import sys
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
count = 0
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

fileflag = sys.argv[1]
outputdir = sys.argv[2]
top_k = int(sys.argv[3])
top_p = float(sys.argv[4])
temp = float(sys.argv[5])
repeat_penalty = float(sys.argv[6])
no_repeat_ngram_size = int(sys.argv[7])
filepath = sys.argv[8]
modelpath = sys.argv[9]

tokenizer = GPT2Tokenizer.from_pretrained(modelpath)
model = GPT2LMHeadModel.from_pretrained(modelpath)
model.eval()
model.to('cuda')

print('outputdir:%s, top_k:%f, top_p:%f, temp:%f, repeat_penalty:%f, no_repeat_ngram_size:%f'%(outputdir, top_k, top_p, temp, repeat_penalty, no_repeat_ngram_size))

if fileflag == "True":
    a = []
    for line in open(filepath):
        a.append(line.split(' ====== ')[0]+' ====== ')
    f = open(outputdir,'w')
    for elem in tqdm(a):
        #elem = elem.replace(' %%%%  ','')
        input_ids = torch.tensor(tokenizer.encode(elem, add_special_tokens=True)).unsqueeze(0).to('cuda')
        #sample_output = model.generate(input_ids,do_sample=True,max_length=150,top_k=20,temperature=0.7,no_repeat_ngram_size=3)
        sample_output = model.generate(input_ids,do_sample=True,max_length=300,top_k=top_k,top_p=top_p,temperature=temp,
            no_repeat_ngram_size=no_repeat_ngram_size, repetition_penalty=repeat_penalty)
        op = tokenizer.decode(sample_output[0].tolist(), skip_special_tokens=True)
        #op = op.split(' $ ')
        #op = [o.capitalize() for o in op]
        #op = '. $ '.join(op)
        #op = post_process(op)
        #op = op.lower()
        send = op.split('======')[1]
        f.write(op+'\n')
        #if count>10:
            #break
        #count = count+1
    f.close()


else:
    elem = sys.argv[2]
    input_ids = torch.tensor(tokenizer.encode(elem, add_special_tokens=True)).unsqueeze(0)
    sample_output = model.generate(input_ids,do_sample=True,max_length=150,top_k=20,temperature=0.7,no_repeat_ngram_size=3)
    op = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    #op = op.split(' $ ')
    #op = [o.capitalize() for o in op]
    #op = '. $ '.join(op)
    #op = post_process(op)
    print(op+'\n')
#print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))