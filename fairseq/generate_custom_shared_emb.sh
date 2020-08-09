#!/usr/bin/env bash
datapath=temp_single_vocab/bpe_single_dict/bin
checkpointdir=checkpoints
modeltypes=(lstm_all_dim_273_attention_share_all lstm_all_dim_273_share_all lstm_emb_273_ffc_512_share_all_attention lstm_emb_273_ffc_512_share_all_non-attention lstm_emb_512_ffc_273_share_all_attention lstm_emb_512_ffc_273_share_all_non-attention transformer_emb_256_ffc_256_fp16_share_all transformer_emb_256_ffc_512_fp16_decoder_shared transformer_emb_256_ffc_512_fp16_share_all)
modelname=checkpoint_best.pt
bpe_type=subword_nmt
bpe_codes=temp_single_vocab/bpe_single_dict/code
sourcedir=temp
source1=conll_unsigned_test.src
source2=test.src
tokenizer=nltk
echo $datapath
echo $bpe_codes
CUDA_VISIBLE_DEVICES=0 
for model in ${modeltypes[@]}; do
for src in $source1 $source2; do
    modeldir=$checkpointdir/$model
    target=inference_result/${model}_${src}.txt
    echo $modeldir
    echo src=$sourcedir/$src
    echo tgt=$target
    python generate_custom.py $datapath $modeldir $modelname $bpe_type $bpe_codes $sourcedir/$src $target $tokenizer
done
done
