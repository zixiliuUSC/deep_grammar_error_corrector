python gec-ranking/scripts/compute_gleu.py \
    -s ../fairseq/temp/conll_unsigned_test.tgt \
    -r ../fairseq/temp/conll_unsigned_test.tgt \
    -o inference_result/gpt2_conll_unsigned_test.txt \
    -n 4