python train.py temp/bpe/bin \
    --save-dir checkpoints/lstm_all_dim_273 --arch lstm \
    --dropout 0.2 \
    --optimizer adam --lr 0.005 --lr-shrink 0.5 \
    --max-tokens 8000 --task translation    \
    --bpe subword_nmt \
    --max-epoch 50 \
    --encoder-embed-dim 273 \
    --decoder-embed-dim 273 \
    --encoder-hidden-size 273 \
    --decoder-hidden-size 273 \
    --encoder-layers 3 \
    --decoder-layers 3 \
    --update-freq 8 \
    --decoder-out-embed-dim 273 \
    --warmup-updates 200 \
    --decoder-attention False \
    --weight-decay 0.1 \
    --fp16 \
    --patience 8 \

#--ddp-backend no_c10d \
#    --keep-best-checkpoints 1 \
#    