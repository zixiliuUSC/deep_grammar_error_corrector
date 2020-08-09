python train.py temp/bpe/bin \
    --save-dir checkpoints/simple_lstm --arch tutorial_simple_lstm \
    --encoder-dropout 0.2 --decoder-dropout 0.2\
    --optimizer adam --lr 0.005 --lr-shrink 0.5 \
    --max-tokens 8000 --task translation    \
    --bpe subword_nmt \
    --max-epoch 50 \
    --encoder-embed-dim 350 \
    --decoder-embed-dim 350 \
    --encoder-hidden-dim 350 \
    --decoder-hidden-dim 350 \
    --update-freq 8 \
    --warmup-updates 200 \
    --weight-decay 0.1 \
    --fp16 \
    --patience 8