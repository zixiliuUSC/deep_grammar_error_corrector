EMBED_SIZE=273
HIDDEN_SIZE=273
CHECKPOINT_PATH=checkpoints/lstm_all_dim_273_attention_share_all
python train.py temp_single_vocab/bpe_single_dict/bin \
    --save-dir $CHECKPOINT_PATH --arch lstm \
    --dropout 0.2 \
    --optimizer adam --lr 0.005 --lr-shrink 0.5 \
    --max-tokens 8000 --task translation    \
    --max-epoch 15 \
    --bpe subword_nmt \
    --encoder-embed-dim $EMBED_SIZE \
    --decoder-embed-dim $EMBED_SIZE \
    --encoder-hidden-size $HIDDEN_SIZE \
    --decoder-hidden-size $HIDDEN_SIZE \
    --decoder-out-embed-dim $EMBED_SIZE \
    --share-all-embeddings \
    --encoder-layers 3 \
    --decoder-layers 3 \
    --update-freq 8 \
    --warmup-updates 200 \
    --decoder-attention True \
    --weight-decay 0.1 \
    --fp16


#    --keep-best-checkpoints 1 \
#    --patience 8 \
#--ddp-backend no_c10d \