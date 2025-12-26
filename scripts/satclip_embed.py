#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate SatCLIP image embeddings for grid cells from Sentinel-2 imagery.
Embeddings are saved as .pt files (no write-back to grids).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rtree import index as rtree_index
import torch

SATCLIP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SATCLIP_ROOT))

from satclip.load import get_satclip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SatCLIP image embeddings for grid cells.")
    parser.add_argument("--ckpt", required=True, help="Path to SatCLIP checkpoint (.ckpt)")
    parser.add_argument("--imagery-dir", required=True, help="Directory with 13-band Sentinel-2 GeoTIFFs")
    parser.add_argument("--grid", required=True, help="Grid GeoJSON path")
    parser.add_argument("--task", required=True, choices=["gdp_pop_builtup", "landuse"], help="Task name for output")
    parser.add_argument("--city", required=True, help="City name (e.g., beijing, shenzhen)")
    parser.add_argument("--output-dir", required=True, help="Output directory for embeddings")
    parser.add_argument("--intermediate-dir", default=None, help="Directory for intermediate files")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick test")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--normalize", action="store_true", default=True, help="L2 normalize embeddings")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable L2 normalization")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def find_rasters(imagery_dir: Path, city: str) -> List[Path]:
    patterns = [
        f"{city}_13band_*.tif",
        f"{city}_*13band*.tif",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(imagery_dir.glob(pattern)))
    return sorted(set(files))


def build_tile_index(paths: List[Path]) -> Tuple[List[Dict], rtree_index.Index]:
    tiles: List[Dict] = []
    idx = rtree_index.Index()
    for i, path in enumerate(paths):
        with rasterio.open(path) as ds:
            bounds = ds.bounds
            tiles.append(
                {
                    "path": str(path),
                    "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                    "crs": ds.crs.to_string() if ds.crs else None,
                    "count": ds.count,
                }
            )
        idx.insert(i, (bounds.left, bounds.bottom, bounds.right, bounds.top))
    return tiles, idx


def open_datasets(tiles: List[Dict]) -> Dict[int, rasterio.DatasetReader]:
    datasets: Dict[int, rasterio.DatasetReader] = {}
    for i, tile in enumerate(tiles):
        datasets[i] = rasterio.open(tile["path"])
    return datasets


def close_datasets(datasets: Dict[int, rasterio.DatasetReader]) -> None:
    for ds in datasets.values():
        ds.close()


def choose_tile(
    point_xy: Tuple[float, float],
    idx: rtree_index.Index,
    tiles: List[Dict],
) -> Optional[int]:
    x, y = point_xy
    for i in idx.intersection((x, y, x, y)):
        left, bottom, right, top = tiles[i]["bounds"]
        if left <= x <= right and bottom <= y <= top:
            return i
    return None


def adjust_channels(image: np.ndarray, in_channels: int) -> np.ndarray:
    if image.shape[0] == in_channels:
        return image
    if image.shape[0] == 12 and in_channels == 13:
        b10 = np.zeros((1, image.shape[1], image.shape[2]), dtype=image.dtype)
        return np.concatenate([image[:10], b10, image[10:]], axis=0)
    if image.shape[0] == 13 and in_channels == 12:
        return np.concatenate([image[:10], image[11:]], axis=0)
    raise ValueError(f"Channel mismatch: image has {image.shape[0]} bands, model expects {in_channels}")


def read_patch(
    ds: rasterio.DatasetReader,
    geom_bounds: Tuple[float, float, float, float],
    out_size: int,
    in_channels: int,
) -> np.ndarray:
    window = rasterio.windows.from_bounds(*geom_bounds, transform=ds.transform)
    data = ds.read(
        window=window,
        out_shape=(ds.count, out_size, out_size),
        resampling=Resampling.bilinear,
        boundless=True,
        fill_value=0,
    )
    data = data.astype(np.float32) / 10000.0
    data = adjust_channels(data, in_channels)
    return data


def load_ckpt_hparams(ckpt_path: Path) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("hyper_parameters", {})


def main() -> None:
    args = parse_args()
    city = args.city.lower()

    imagery_dir = Path(args.imagery_dir)
    grid_path = Path(args.grid)
    ckpt_path = Path(args.ckpt)
    output_dir = Path(args.output_dir)
    if args.intermediate_dir:
        intermediate_dir = Path(args.intermediate_dir)
    else:
        intermediate_dir = output_dir / "intermediate"

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    hparams = load_ckpt_hparams(ckpt_path)
    in_channels = int(hparams.get("in_channels", 13))
    image_resolution = int(hparams.get("image_resolution", 256))
    embed_dim = int(hparams.get("embed_dim", 512))

    raster_paths = find_rasters(imagery_dir, city)
    if not raster_paths:
        raise FileNotFoundError(f"No imagery found for city '{city}' in {imagery_dir}")

    tiles, idx = build_tile_index(raster_paths)
    datasets = open_datasets(tiles)
    raster_crs = datasets[0].crs

    gdf = gpd.read_file(grid_path)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    if (~gdf.is_valid).any():
        gdf.loc[~gdf.is_valid, "geometry"] = gdf.loc[~gdf.is_valid, "geometry"].buffer(0)

    ids = gdf["id"].astype(str).tolist() if "id" in gdf.columns else gdf.index.astype(str).tolist()
    centroids = gdf.geometry.centroid

    max_samples = args.max_samples or len(gdf)
    max_samples = min(max_samples, len(gdf))

    device = select_device(args.device)
    model = get_satclip(str(ckpt_path), device=device, return_all=True)
    model.eval()

    embeddings: List[torch.Tensor] = []
    kept_ids: List[str] = []
    kept_centroids: List[Tuple[float, float]] = []
    skipped_ids: List[str] = []

    batch: List[torch.Tensor] = []
    batch_ids: List[str] = []
    batch_centroids: List[Tuple[float, float]] = []

    with torch.no_grad():
        for i in range(max_samples):
            geom = gdf.geometry.iloc[i]
            geom_bounds = geom.bounds
            point = centroids.iloc[i]
            tile_id = choose_tile((point.x, point.y), idx, tiles)
            if tile_id is None:
                skipped_ids.append(ids[i])
                continue

            ds = datasets[tile_id]
            patch = read_patch(ds, geom_bounds, image_resolution, in_channels)
            tensor = torch.from_numpy(patch)
            batch.append(tensor)
            batch_ids.append(ids[i])
            batch_centroids.append((float(point.x), float(point.y)))

            if len(batch) >= args.batch_size:
                imgs = torch.stack(batch).to(device)
                emb = model.encode_image(imgs).float()
                if args.normalize:
                    emb = emb / emb.norm(dim=1, keepdim=True)
                embeddings.append(emb.cpu())
                kept_ids.extend(batch_ids)
                kept_centroids.extend(batch_centroids)
                batch, batch_ids, batch_centroids = [], [], []

        if batch:
            imgs = torch.stack(batch).to(device)
            emb = model.encode_image(imgs).float()
            if args.normalize:
                emb = emb / emb.norm(dim=1, keepdim=True)
            embeddings.append(emb.cpu())
            kept_ids.extend(batch_ids)
            kept_centroids.extend(batch_centroids)

    close_datasets(datasets)

    if embeddings:
        embeddings_tensor = torch.cat(embeddings, dim=0)
    else:
        embeddings_tensor = torch.empty((0, embed_dim), dtype=torch.float32)

    torch.save(
        {
            "ids": kept_ids,
            "centroids": kept_centroids,
            "embeddings": embeddings_tensor,
            "task": args.task,
            "city": city,
            "grid_path": str(grid_path),
            "ckpt_path": str(ckpt_path),
            "image_resolution": image_resolution,
            "in_channels": in_channels,
            "embed_dim": embed_dim,
            "normalize": args.normalize,
        },
        output_dir / "embeddings.pt",
    )

    meta = {
        "task": args.task,
        "city": city,
        "grid_path": str(grid_path),
        "imagery_dir": str(imagery_dir),
        "ckpt_path": str(ckpt_path),
        "image_resolution": image_resolution,
        "in_channels": in_channels,
        "embed_dim": embed_dim,
        "normalize": args.normalize,
        "num_samples": len(gdf),
        "num_embedded": len(kept_ids),
        "num_skipped": len(skipped_ids),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if skipped_ids:
        with (intermediate_dir / "skipped_ids.json").open("w", encoding="utf-8") as f:
            json.dump(skipped_ids, f, indent=2)

    tile_index_path = intermediate_dir / "tile_index.json"
    with tile_index_path.open("w", encoding="utf-8") as f:
        json.dump(tiles, f, indent=2)

    print(f"Saved embeddings to: {output_dir / 'embeddings.pt'}")
    print(f"Saved meta to: {output_dir / 'meta.json'}")
    if skipped_ids:
        print(f"Skipped {len(skipped_ids)} samples (see skipped_ids.json).")


if __name__ == "__main__":
    main()
