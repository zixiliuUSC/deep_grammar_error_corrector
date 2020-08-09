This repo aims at comparing Seq2Seq model using different neural architecture in correcting grammar. Currently our models include RNN-based, LSTM-based, transformer-based models. 
### Requirement and installation:
## for training and generation
python: 3.7.6
pytorch: 1.4.0
CUDA: 10.1
subword-nmt: 0.3.7
nltk: 3.4.5

FairSeq installation:
```
cd fairseq
pip install --editable .
```

Huggingface transformers installation:
```
cd transformers
pip install .
```
## for evaluation
python: 2.7.18
numpy: 1.16.6

### FairSeq preprocessing:
## non-shared dictionary:
```
cd temp
sh create_bpe.sh
cd ..
sh preprocess.sh
```
## shared dictionary
```
cd temp_single_vocab
sh create_bpe_single_dict.sh
cd ..
sh preprocess_sigle_dict.sh
```

### train model 1-14:
```
cd fairseq
```
# model 1
```sh train_simple_lstm.sh```
# model 2
```sh train_lstm.sh```
# model 3 
```sh train_lstm_attention.sh```
# model 4,6,8
adjust following things in `train_lstm_single_dict.sh`: adjust embedding size and hidden size according to model config in report paper and  checkpoint path as that in `checkpoints` folder
```sh train_lstm_single_dict.sh```
# model 5,7,9
adjust following things in `train_lstm_attention_single_dict.sh`: adjust embedding size and hidden size according to model config in report paper and  checkpoint path as that in `checkpoints` folder
```sh train_lstm_attention_single_dict.sh```
# model 10,11
adjust following things in `train_transformer.sh`: adjust hidden size according to model config in report paper and  checkpoint path as that in `checkpoints` folder
```sh train_transformer.sh```
# model 12,13
adjust following things in `train_transformer_single_dict.sh`: adjust hidden size according to model config in report paper and  checkpoint path as that in `checkpoints` folder
```sh train_transformer_single_dict.sh```
# model 14
```train_transformer_single_decoder_dict.sh```
### model 15
```
cd transformer
sh train_gpt2.sh
```
### inference
For model 1-14, our script will generate 2 inference result for each model, one for Conll 2014 test set, one for Conll 2014+lang-8 test set.
For model 15, our script only generate inference result for Conll 2014 test set.
In this zip file, we only keep the best check point of model 13 which is the best model in this project. 
If you want to check out other checkpoints, please download from this website 
## model 4, 5, 6, 7, 8, 9, 12, 13, 14
```
sh generate_custom_shared_emb.sh
```
## model 1, 2, 3, 10, 11
```
sh generate_custom_non-shared.sh
```
## model 15
```
cd transformer
sh generate_gpt2.sh
```
### evaluation
## model 1-14
configure `Conll` and `lang8_conll` in `gleu_eval.sh`
then run `cd fariseq;sh gleu_eval.sh`, it will print print score for each file. 
directly run `gleu_eval.sh` under `fairseq`, you will get the evaluation score of model 13 in the paper. 
## model 15 
```
cd transformers
sh gleu_eval.sh
```
