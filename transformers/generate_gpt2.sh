fileflag=True
outputdir=inference_result/gpt2_conll_unsigned_test.txt
top_k=5
top_p=0.9
temp=0.7
repeat_penalty=1
no_repeat_ngram_size=3
filepath=../fairseq/temp/conll_unsigned_test.src
modelpath=checkpoints/checkpoint-3000

python generate_gpt2.py $fileflag $outputdir $top_k $top_p $temp $repeat_penalty $no_repeat_ngram_size $filepath $modelpath

#story_top-k_5_top-p_0.9_temp_0.7_repeat-penalty_1_no-ngram-size_3.txt