#!/usr/bin/env bash
set -euo pipefail

# Full run (adjust CKPT if needed)
CKPT="weights/satclip-vit16-l40.ckpt"
IMAGERY_DIR="../../Data/Sentinel2_L1C"

python scripts/satclip_embed.py \
  --ckpt "${CKPT}" \
  --imagery-dir "${IMAGERY_DIR}" \
  --grid "../../Data/Grid/grid_gdp_pop_builtup/beijing_grid_gdp_pop.geojson" \
  --task "gdp_pop_builtup" \
  --city "beijing" \
  --output-dir "outputs/satclip/embeddings/gdp_pop_builtup/beijing" \
  --intermediate-dir "outputs/satclip/intermediate/gdp_pop_builtup/beijing" \
  --batch-size 16

python scripts/satclip_embed.py \
  --ckpt "${CKPT}" \
  --imagery-dir "${IMAGERY_DIR}" \
  --grid "../../Data/Grid/grid_gdp_pop_builtup/shenzhen_grid_gdp_pop.geojson" \
  --task "gdp_pop_builtup" \
  --city "shenzhen" \
  --output-dir "outputs/satclip/embeddings/gdp_pop_builtup/shenzhen" \
  --intermediate-dir "outputs/satclip/intermediate/gdp_pop_builtup/shenzhen" \
  --batch-size 16

python scripts/satclip_embed.py \
  --ckpt "${CKPT}" \
  --imagery-dir "${IMAGERY_DIR}" \
  --grid "../../Data/Grid/grid_lu/beijing_grid_lu.geojson" \
  --task "landuse" \
  --city "beijing" \
  --output-dir "outputs/satclip/embeddings/landuse/beijing" \
  --intermediate-dir "outputs/satclip/intermediate/landuse/beijing" \
  --batch-size 16

python scripts/satclip_embed.py \
  --ckpt "${CKPT}" \
  --imagery-dir "${IMAGERY_DIR}" \
  --grid "../../Data/Grid/grid_lu/shenzhen_grid_lu.geojson" \
  --task "landuse" \
  --city "shenzhen" \
  --output-dir "outputs/satclip/embeddings/landuse/shenzhen" \
  --intermediate-dir "outputs/satclip/intermediate/landuse/shenzhen" \
  --batch-size 16
