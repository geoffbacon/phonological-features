#!/bin/bash

set -euo pipefail

LG=$1
DATA_DIR=data/word/wiki40b/$LG
MODEL_DIR=models/word/$LG/transformer
echo Preprocessing
# fairseq-preprocess \
#     --trainpref $DATA_DIR/train.txt \
#     --validpref $DATA_DIR/validation.txt \
#     --testpref $DATA_DIR/test.txt \
#     --destdir $MODEL_DIR/bin \
#     --only-source \
#     --tokenizer space \
#     --thresholdsrc 5 \
#     --workers 1 \
#     --cpu
echo Training
# fairseq-train $MODEL_DIR/bin \
#     --save-dir $MODEL_DIR \
#     --task language_modeling \
#     --arch transformer_lm \
#     --activation-fn gelu \
#     --dropout 0.1 \
#     --attention-dropout 0.1 \
#     --decoder-embed-dim 100 \
#     --decoder-layers 2 \
#     --decoder-attention-heads 2 \
#     --share-decoder-input-output-embed \
#     --optimizer adam \
#     --adam-betas '(0.9, 0.98)' \
#     --weight-decay 0.01 \
#     --clip-norm 0.0 \
#     --lr 0.0005 \
#     --lr-scheduler inverse_sqrt \
#     --warmup-updates 10 \
#     --warmup-init-lr 1e-07 \
#     --tokens-per-sample 512 \
#     --max-tokens 512 \
#     --sample-break-mode none \
#     --max-update 5 \
#     --seed 2020 \
#     --cpu

# --tokenizer space
# --bpe fastbpe
# --fp16

echo Evaluating
fairseq-eval-lm $MODEL_DIR/bin \
    --path $MODEL_DIR/checkpoint_best.pt \
    --sample-break-mode complete \
    --max-tokens 3072 \
    --context-window 1000 \
    --softmax-batch 8