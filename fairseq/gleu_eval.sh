Conll=inference_result/transformer_emb_256_ffc_512_fp16_share_all_conll_unsigned_test.src.txt
lang8_conll=inference_result/transformer_emb_256_ffc_512_fp16_share_all_test.src.txt
python gec-ranking/scripts/compute_gleu.py \
    -s temp/conll_unsigned_test.tgt \
    -r temp/conll_unsigned_test.tgt \
    -o $Conll \
    -n 4
python gec-ranking/scripts/compute_gleu.py \
    -s temp/test.tgt \
    -r temp/test.tgt \
    -o $lang8_conll \
    -n 4
