#!/usr/bin/env bash
set -euo pipefail

# Small-batch smoke test (adjust CKPT if needed)
CKPT="weights/satclip-vit16-l40.ckpt"
IMAGERY_DIR="../../Data/Sentinel2_L1C"

python scripts/satclip_embed.py \
  --ckpt "${CKPT}" \
  --imagery-dir "${IMAGERY_DIR}" \
  --grid "../../Data/Grid/grid_gdp_pop_builtup/beijing_grid_gdp_pop.geojson" \
  --task "gdp_pop_builtup" \
  --city "beijing" \
  --output-dir "outputs/satclip/embeddings/gdp_pop_builtup/beijing_test" \
  --intermediate-dir "outputs/satclip/intermediate/gdp_pop_builtup/beijing_test" \
  --batch-size 2 \
  --max-samples 8

python scripts/satclip_embed.py \
  --ckpt "${CKPT}" \
  --imagery-dir "${IMAGERY_DIR}" \
  --grid "../../Data/Grid/grid_lu/shenzhen_grid_lu.geojson" \
  --task "landuse" \
  --city "shenzhen" \
  --output-dir "outputs/satclip/embeddings/landuse/shenzhen_test" \
  --intermediate-dir "outputs/satclip/intermediate/landuse/shenzhen_test" \
  --batch-size 2 \
  --max-samples 8
