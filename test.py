import os
import sys
from argparse import ArgumentParser

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import Config
from model import TextMappingModel
from data import LPMappingDataset
from data_per_declaration import DeclarationMappingDataset
from constants import SPECIAL_TOKENS
from utils import *
import test_utils

# configuration
parser = ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
# parser.add_argument('--eval-train', action="store_true", dest="eval_train")
# parser.add_argument('--no-eval-train', action="store_false", dest="eval_train")
parser.add_argument('--test-file', type=str, default="")
parser.add_argument('--beam-size', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--outfilename', type=str, default='results.out')
# parser.add_argument('--debug', action="store_true", dest="debug")
# parser.add_argument('--no-debug', action="store_false", dest="debug")

# parser.set_defaults(eval_train=False)
parser.set_defaults(debug=False)

args = parser.parse_args()

use_gpu = args.gpu > -1
checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpu}' if use_gpu else 'cpu')
config = Config.from_dict(checkpoint['config'])
config.eval_batch_size = args.batch_size

# increase recursion limit
max_length = int(config.max_length)
sys.setrecursionlimit(max_length  * max_length + 100)

# Override test file
if args.test_file:
    config.test_file = args.test_file

# set GPU device
config.gpu_device = args.gpu
config.use_gpu = use_gpu

if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

print(config)

# datasets
model_name = config.bert_model_name

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir)
tokenizer.add_tokens(SPECIAL_TOKENS)

# Output result files
ckpt_path_splits = args.checkpoint.split('/')
ckpt_basename = ckpt_path_splits[-1].split('.')[0]
if args.debug:
    ckpt_basename = 'debug-' + ckpt_basename

output_test_prefix = os.path.basename(args.test_file).split('.')[0] if args.test_file else "test"
output_dir = '/'.join(ckpt_path_splits[:-1])
train_result_file = os.path.join(output_dir, 'results', ckpt_basename, 'train.out.json')
dev_result_file = os.path.join(output_dir, 'results', ckpt_basename, 'dev.out.json')
test_result_file = os.path.join(output_dir, 'results', ckpt_basename, '{}.out.json'.format(output_test_prefix))
train_score_file = os.path.join(output_dir, 'results', ckpt_basename, 'train.score.json')
dev_score_file = os.path.join(output_dir, 'results', ckpt_basename, 'dev.score.json')
test_score_file = os.path.join(output_dir, 'results', ckpt_basename, '{}.score.json'.format(output_test_prefix))

os.makedirs(os.path.join(output_dir, 'results', ckpt_basename), exist_ok=True)

train_set = None

print('============== Prepare Test Set: starting =================')
if config.per_declaration:
    test_set = DeclarationMappingDataset(config.test_file, tokenizer=tokenizer, max_length=config.max_length, gpu=use_gpu, enrich_ner=config.enrich_ner)
else:
    test_set = LPMappingDataset(config.test_file, tokenizer=tokenizer, max_length=config.max_length, gpu=use_gpu, enrich_ner=config.enrich_ner)
test_set.numberize()
print('============== Prepare Test Set: finished =================')

# initialize the model
model = TextMappingModel(config)
model.load_bert(model_name, cache_dir=config.bert_cache_dir, tokenizer=tokenizer)
if not model_name.startswith('roberta'):
    model.bert.resize_token_embeddings(len(tokenizer))
model.load_state_dict(checkpoint['model'], strict=True)

if use_gpu:
    model.cuda(device=config.gpu_device)
epoch = 1000

# Number of batches
batch_num = len(test_set) // config.eval_batch_size + \
                (len(test_set) % config.eval_batch_size != 0)

# Test set
test_result = test_utils.evaluate(
        tokenizer,
        model,
        test_set,
        epoch,
        batch_num,
        use_gpu,
        config,
        tqdm_descr='Test',
        ckpt_basename=ckpt_basename,
        beam_size=args.beam_size,
        per_declaration=config.per_declaration,
    )

print(f'Accuracy: {test_result["accuracy"]}')

with open(test_result_file, 'w') as f:
    f.write(json.dumps(test_result))

# output result to file for evaluation
outfilepath = os.path.join(output_dir, args.outfilename)
with open(outfilepath, 'w') as f:
    output = {
        "accuracy": test_result["accuracy"],
        "seed": config.seed,
        "test_file": args.test_file,
        "beam_size": args.beam_size,
    }
    f.write(json.dumps(output, indent = 4))
