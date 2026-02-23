from argparse import ArgumentParser
import os
import torch

EMBEDDING_MODEL_INSTRUCTIONS = {
    'Qwen/Qwen3-Embedding-0.6B': 'Instruct: Retrieve relevant news articles based on user browsing history\nQuery:',
    'intfloat/e5-small-v2': ['query: ', 'passage: '],
    'intfloat/e5-base-v2': ['query: ', 'passage: '],
    'intfloat/e5-large-v2': ['query: ', 'passage: '],
}

def _comma_list(value):
    l = value.split(',')
    return l

def parse_args():
    parser = ArgumentParser()
    # path and file
    parser.add_argument("--base_dir", type=str, default=os.getcwd())
    parser.add_argument("--pretrain_ckpt", type=str, default=None, required=False)
    parser.add_argument("--data_path", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt", type=str, default="best_model.bin")
    parser.add_argument("--hf_dir", type=str, default="hf_repo")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--train_file", type=str, default="train.parquet")
    parser.add_argument("--dev_file", type=str, default="dev.parquet")
    parser.add_argument("--test_file", type=str, default="test.parquet")
    parser.add_argument("--item2id_file", type=_comma_list, default=["smap.json"]) # label-to-id
    parser.add_argument("--impression2id_file", type=str, default='imap.json')
    parser.add_argument("--meta_file", type=_comma_list, default=["meta_data.json"]) # label-to-attr
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ext_val_data_path", type=_comma_list, default=None)
    parser.add_argument("--ext_val_meta_file", type=_comma_list, default=None) # label-to-attr
    parser.add_argument("--unified", action='store_true')
    parser.add_argument("--source_ref", type=_comma_list, default=['mind_small'])

    # data process
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int, default=8, help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_multiprocessing", action='store_true')

    # model
    parser.add_argument("--temp", type=float, default=0.05, help="Temperature for softmax.")
    parser.add_argument("--freeze_temp", action='store_false')
    parser.add_argument("--max_attr_num", type=int, default=3, help="Max item attributes")
    parser.add_argument("--max_attr_length", type=int, default=128, help="Max item attributes length")
    parser.add_argument("--max_token_num", type=int, default=2048, help="Max sequence token length")
    parser.add_argument("--max_item_embeddings", type=int, default=51, help="Max items in history")
    parser.add_argument(
        "--token_pooling", type=str, choices=['mean', 'cls', 'last'], default='mean'
    )
    parser.add_argument(
        "--item_pooling", type=str, 
        choices=['mean', 'cls', 'last', 'sas', 'topk'], default='mean'
    )
    parser.add_argument("--item_pool_prenorm", action='store_true')
    parser.add_argument("--pooling_topk", type=int, default=None)   
    # train
    parser.add_argument("--loss", 
        choices=['bce_temp', 'cos_emb', 'margin_rank', 'infonce', 'ranknet', 'multi_infonce']
    )
    parser.add_argument("--uniformity_lam", type=float, default=None)
    parser.add_argument("--score_redun_lam", type=float, default=None) # account for fact that larger max cosine scaled with temp will be much larger than sim cosine
    parser.add_argument("--redun_sim_thresh", type=float, default=None)
    parser.add_argument("--contrast_margin", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--finetune_negative_sample_size", type=int, default=1000)
    parser.add_argument("--train_num_candidates_per_row", type=int, default=2)
    parser.add_argument("--train_multiple_positives", action='store_true')
    parser.add_argument("--disable_item_sampling", action='store_true')
    parser.add_argument(
        "--validation_steps", type=int, default=100, help='ideally > 50, past warmup'
    )
    parser.add_argument('--update_embeddings_steps', type=int, default=None)
    parser.add_argument("--mini_validation_size", type=int, default=1024)
    parser.add_argument(
        "--metric_ks", nargs="+", type=int, default=[5, 10], help="ks for Metric@k"
    )
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--sas_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=2.0)
    parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fix_word_embedding", action="store_true")
    parser.add_argument("--use_item_table", action="store_true")
    parser.add_argument("--freeze_lm", action="store_true")
    parser.add_argument("--unfreeze_item_embeddings", action="store_true")
    parser.add_argument("--item_pad_id", type=int, default=0)
    parser.add_argument("--use_score_head", action='store_true')
    parser.add_argument("--score_mix_lam", type=float, default=None)
    parser.add_argument("--lora", action='store_true')
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_bias", type=str, choices=['all', 'none'], default='all')

    parser.add_argument("--verbose", type=int, default=3)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    # other benchmarking arguments
    parser.add_argument(
        "--encoder_method", type=str, default=None, required=False,
        choices=['bi', 'cross', 'bi_colbert', 'cross_colbert', 'full_cross', 'bi_mix_topk_mean'],
        help='Encoder scoring method for benchmarking'
    )
    parser.add_argument("--padding_side", type=str, default=None, help="Tokenizer padding side")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Attention implementation for model")
    parser.add_argument("--compile", action='store_true', help="torch.compile on model")
    parser.add_argument(
        "--symmetric", action='store_true',
        help="Whether to use instructions for both history and candidates"
    )

    args = parser.parse_args()
    # checks
    assert not args.unified or (
        isinstance(args.meta_file, list) and isinstance(args.item2id_file, list) \
        and len(args.meta_file) == len(args.item2id_file) \
        and len(args.item2id_file) == len(args.source_ref)
    ), f'{args.meta_file}, {args.item2id_file}, {args.source_ref}'
    assert not args.loss == 'multi_infonce' or args.train_multiple_positives
    assert not args.encoder_method == 'full_cross' or args.use_score_head
    assert not args.encoder_method == 'bi_mix_topk_mean' \
            or args.item_pooling in ['topk' ]
    if args.loss not in ['bce_temp', 'infonce', 'multi_infonce', 'ranknet']:
        import warnings; warnings.warn(f'Temperature scaling not used for logis for {args.loss} Loss')
    
    args.data_path = os.path.join(args.base_dir, args.data_path)
    args.output_dir = os.path.join(args.base_dir, args.output_dir)
    args.log_dir = os.path.join(args.base_dir, args.log_dir)
    
    args.device = (
        torch.device("cuda:{}".format(args.device))
        if args.device >= 0
        else torch.device("cpu")
    )
    args.dtype = torch.bfloat16 if args.bf16 else torch.float32
    args.instruction = EMBEDDING_MODEL_INSTRUCTIONS.get(
        args.model_name_or_path,
        'Capture taste and match more of this text: ',
    )
    
    if args.debug:
        args.num_train_epochs = max(10, args.num_train_epochs)
        args.early_stopping_rounds_stage_1 = 99
        args.early_stopping_rounds_stage_2 = 99
        args.gradient_accumulation_steps = 1
        args.mini_validation_size = None
        args.warmup_steps = 0
        args.log_steps = 1
        args.dataloader_multiprocessing = False
        args.compile = False
    if args.profile: args.gradient_accumulation_steps = 1
        
    for k, v in vars(args).items(): print(f"  {k:30}: {v}")
    return args