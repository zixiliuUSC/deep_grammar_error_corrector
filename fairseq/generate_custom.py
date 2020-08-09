import torch 
import sys
from fairseq.models.transformer import TransformerModel
#import argparse
#parser = argparse.ArgumentParser()


datapath = sys.argv[1]
modeldir = sys.argv[2]
modelname = sys.argv[3]
bpe_type = sys.argv[4]
bpe_codes = sys.argv[5]
source = sys.argv[6]
target = sys.argv[7]
tokenizer = sys.argv[8]

#parser.add_argument('--tokenizer',default='nltk')
#args = parser.parse_args(['--tokenizer',tokenizer])

model = TransformerModel.from_pretrained(
    modeldir,
    checkpoint_file=modelname,
    data_name_or_path=datapath,
    bpe=bpe_type,
    bpe_codes=bpe_codes,
    tokenizer=tokenizer,max_target_positions=1500 # this argument is **kwargs, you can allow archive_map to set default arg_overrides (e.g., tokenizer, bpe), for each model
    )
model.cuda()
model.eval()
model.half()
count = 1
bsz = 500
with open(source,'r') as source, open(target, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = model.sample(slines, beam=4, lenpen=0.9, max_len_b=150, min_len=1, no_repeat_ngram_size=3,unkpen=0.5)

            for hypothesis in hypotheses_batch:
                hypothesis = hypothesis.replace('  ',' ')
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = model.sample(slines, beam=4, lenpen=0.9, max_len_b=150, min_len=1, no_repeat_ngram_size=3,unkpen=0.5)
        for hypothesis in hypotheses_batch:
            hypothesis = hypothesis.replace('  ',' ')
            fout.write(hypothesis + '\n')
            fout.flush()
'''
TransformerModel.from_pretrained(cls,model_name_or_path,checkpoint_file="model.pt",data_name_or_path=".",**kwargs)
**kwargs has all the namespace as below. 
Namespace(activation_dropout=0.1, activation_fn='relu', adam_betas='(0.9, 0.999)', adam_eps=1e-08, adaptive_input=False, adaptive_softmax_cutoff=None, 
adaptive_softmax_dropout=0, all_gather_list_size=16384, arch='transformer', attention_dropout=0.1, best_checkpoint_metric='loss', bpe='subword_nmt', 
bpe_codes='temp/bpe/code', bpe_separator='@@', broadcast_buffers=False, bucket_cap_mb=25, checkpoint_suffix='', clip_norm=25, cpu=False, 
criterion='cross_entropy', cross_self_attention=False, curriculum=0, data='temp/bpe/bin', data_buffer_size=0, dataset_impl=None, ddp_backend='c10d', 
decoder_attention_heads=4, decoder_embed_dim=256, decoder_embed_path=None, decoder_ffn_embed_dim=256, decoder_input_dim=256, decoder_layerdrop=0, 
decoder_layers=3, decoder_layers_to_keep=None, decoder_learned_pos=False, decoder_normalize_before=True, decoder_output_dim=256, device_id=0, 
disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_port=-1, 
distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=4, 
encoder_embed_dim=256, encoder_embed_path=None, encoder_ffn_embed_dim=256, encoder_layerdrop=0, encoder_layers=3, encoder_layers_to_keep=None, 
encoder_learned_pos=False, encoder_normalize_before=True, eval_bleu=False, eval_bleu_args=None, eval_bleu_detok='space', eval_bleu_detok_args=None, 
eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, 
fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=True, fp16_init_scale=128, fp16_no_flatten_grads=False, 
fp16_scale_tolerance=0.0, fp16_scale_window=None, keep_best_checkpoints=1, keep_interval_updates=-1, keep_last_epochs=-1, layer_wise_attention=False, 
layernorm_embedding=False, left_pad_source=False, left_pad_target=False, load_alignments=False, localsgd_frequency=3, log_format=None, 
log_interval=100, lr=[0.005], lr_scheduler='fixed', lr_shrink=0.5, max_epoch=0, max_sentences=None, max_sentences_valid=None, max_source_positions=1024, 
max_target_positions=1024, max_tokens=8000, max_tokens_valid=8000, max_update=0, maximize_best_checkpoint_metric=False, memory_efficient_fp16=False, 
min_loss_scale=0.0001, min_lr=-1, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, 
no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_token_positional_embeddings=False, nprocs_per_node=1, 
num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=8, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, 
quantization_config_path=None, required_batch_size_multiple=8, reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, 
restore_file='checkpoint_last.pt', save_dir='checkpoints/transformer_new_fp16', save_interval=1, save_interval_updates=0, seed=1, sentence_avg=False, 
share_all_embeddings=False, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', 
slowmo_momentum=None, source_lang='src', target_lang='tgt', task='translation', tensorboard_logdir='', threshold_loss_scale=None, tokenizer='nltk', 
train_subset='train', truncate_source=False, update_freq=[8], upsample_primary=1, use_bmuf=False, use_old_adam=False, user_dir=None, valid_subset='valid', 
validate_interval=1, warmup_updates=0, weight_decay=0.1)
'''

'''
model.sample(slines, beam=4, lenpen=2.0, max_len_b=300, min_len=55, no_repeat_ngram_size=3,**kwargs)
**kwargs have following argument and some options maybe miss but cab be found from official doc. 
the defining function is in fairseq/tasks/fairseq_task.py def build_generator()
return seq_gen_cls(
    models,
    self.target_dictionary,
    beam_size=getattr(args, "beam", 5),
    max_len_a=getattr(args, "max_len_a", 0),
    max_len_b=getattr(args, "max_len_b", 200),
    min_len=getattr(args, "min_len", 1),
    normalize_scores=(not getattr(args, "unnormalized", False)),
    len_penalty=getattr(args, "lenpen", 1),
    unk_penalty=getattr(args, "unkpen", 0),
    temperature=getattr(args, "temperature", 1.0),
    match_source_len=getattr(args, "match_source_len", False),
    no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
    search_strategy=search_strategy,
)
'''