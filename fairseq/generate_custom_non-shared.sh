#!/usr/bin/env bash
datapath=temp/bpe/bin
checkpointdir=checkpoints
# type folder name in checkpoints folder to modeltypes
modeltypes=(lstm_all_dim_273 lstm_all_dim_273_attention simple_lstm transformer_emb_256_ffc_256_fp16_non-shared transformer_emb_256_ffc_512_fp16_non-share)
modelname=checkpoint_best.pt
bpe_type=subword_nmt
bpe_codes=temp/bpe/code
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


