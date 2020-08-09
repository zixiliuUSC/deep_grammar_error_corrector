HIDDEN_SIZE=256
CHECKPOINT_PATH=checkpoints/transformer_emb_256_ffc_256_fp16
python train.py temp/bpe/bin \
    --save-dir $CHECKPOINT_PATH --arch transformer \
    --activation-fn relu \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
    --encoder-embed-dim 256 \
    --encoder-ffn-embed-dim $HIDDEN_SIZE --encoder-layers 3 \
    --encoder-attention-heads 4 \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim $HIDDEN_SIZE \
    --decoder-layers 3 \
    --decoder-attention-heads 4 \
    --optimizer adam --lr 0.005 --lr-shrink 0.5 \
    --max-tokens 8000 --task translation        \
    --keep-best-checkpoints 1 \
    --bpe subword_nmt \
    --update-freq 8 \
    --patience 8 \
    --best-checkpoint-metric loss \
    --decoder-normalize-before --encoder-normalize-before \
    --weight-decay 0.1 --fp16 \
