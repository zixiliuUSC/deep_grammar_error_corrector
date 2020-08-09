python preprocess.py --source-lang src --target-lang tgt \
    --trainpref temp_single_vocab/bpe_single_dict/train.bpe --validpref temp_single_vocab/bpe_single_dict/val.bpe --testpref temp_single_vocab/bpe_single_dict/test.bpe \
    --destdir temp_single_vocab/bpe_single_dict/bin --workers 20 \
    --joined-dictionary
    
