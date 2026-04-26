#!/bin/bash

TYPE="$1"
EPOCH="$2"
CHECKPOINT_DIR="models/$TYPE"
MODALS="tquvz_t2m_sst_msl_topo_ifs"

python3 evaluation_DiG.py \
  --save "$CHECKPOINT_DIR" \
  --test_epoch $EPOCH \
  --pre_milestone $EPOCH \
  --multi_modals "$MODALS"
