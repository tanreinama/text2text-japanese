import argparse
import os
import json
from tqdm import tqdm
import pickle
import numpy as np
import jaconv
import re
from copy import copy
import tensorflow.compat.v1 as tf

from sampling import sample_sequence
from encode_bpe import BPEEncoder_ja
import model

if int(tf.__version__[0]) > 1:
    from model import HParams as HParams
else:
    from tensorflow.contrib.training import HParams


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="source dir", required=True )
    parser.add_argument("--model", help="moddel name", required=True )
    parser.add_argument('--max_answer_len', default=0, type=int, help='Maximum number of answer tokens')
    parser.add_argument('--min_answer_len', default=0, type=int, help='Minimum number of answer tokens')
    parser.add_argument('--train_type', default='QtoA', help='Direction to generate text.')
    parser.add_argument("--dataset_type", help="dataset type", default='split' )
    parser.add_argument("--split_tag", help="text split tag", default='<|SP_QA|>' )
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument("--gpu", help="use gpu number", default='0' )
    parser.add_argument("--verbose", action='store_true' )
    args = parser.parse_args()

    with open('ja-bpe.txt', encoding='utf-8') as f:
        bpe = f.read().split('\n')
    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())
    enc = BPEEncoder_ja(bpe, emoji)
    n_vocab = len(enc)
    eot_token = enc.encode('<|endoftext|>')[0]
    sep_token = enc.encode('<|byte0|>')[0]
    temperature=args.temperature
    top_k=args.top_k
    top_p=args.top_p
    min_answer_len=args.min_answer_len

    if os.path.isfile(args.model+'/hparams.json'):
        with open(args.model+'/hparams.json') as f:
            params = json.loads(f.read())
            hparams = HParams(**params)
            n_prediction = params['n_prediction']
    elif 'small' in args.model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 768,
          "n_head": 12,
          "n_layer": 12
        })
        n_prediction = args.max_answer_len
    elif 'medium' in args.model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 1024,
          "n_head": 16,
          "n_layer": 24
        })
        n_prediction = args.max_answer_len
    elif 'large' in args.model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 1280,
          "n_head": 20,
          "n_layer": 36
        })
        n_prediction = args.max_answer_len
    else:
        raise ValueError('invalid model name.')

    n_contexts = hparams.n_ctx - n_prediction
    if n_prediction <= 0:
        raise ValueError('invalid hparams.json or no --max_answer_len argument.')

    config = tf.ConfigProto()
    if int(args.gpu) >= 0:
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = args.gpu
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        output = sample_sequence(
            hparams=hparams, length=hparams.n_ctx,
            min_length=min_answer_len, context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(args.model)
        saver.restore(sess, ckpt)

        rouge = []
        for curDir, dirs, files in os.walk(args.src_dir):
            for file in tqdm(files):
                if file.endswith(".txt"):
                    input = os.path.join(curDir, file)
                    with open(input, 'r', encoding='utf-8') as fp:
                        raw_text = fp.read()
                        if args.dataset_type == 'livedoor':
                            raw_line = raw_text.split('\n')
                            tokens = [enc.encode(raw_line[2]),enc.encode('\n'.join(raw_line[3:]))]
                        elif args.dataset_type == 'split':
                            tk = raw_text.split(args.split_tag)
                            tokens = [enc.encode(tk[0]),enc.encode(tk[1])]
                        else:
                            raise ValueError('invalid data name.')
                        if args.train_type == 'QtoA':
                            context_tokens = tokens[1][:n_contexts] + [sep_token]
                            true_tokens = tokens[0]
                        elif args.train_type == 'AtoQ':
                            context_tokens = tokens[0][:n_contexts] + [sep_token]
                            true_tokens = tokens[1]
                        else:
                            raise ValueError('invalid type name.')
                        out = sess.run(output, feed_dict={
                            context: [context_tokens]
                        })[:,len(context_tokens):hparams.n_ctx]
                        pred_tokens = []
                        for wd in out[0]:
                            if wd == eot_token:
                                break
                            pred_tokens.append(wd)
                        f1 = 0
                        if len(pred_tokens) > 0:
                            inter = set(pred_tokens) & set(true_tokens)
                            ninter = len(inter)
                            recall = ninter / len(pred_tokens)
                            precision = ninter / len(true_tokens)
                            f1 = 2 * precision * recall / (precision + recall + 1e-10)
                            if args.verbose:
                                print(f'{f1:.2f} {raw_line[2]} --> {enc.decode(pred_tokens)}')
                        rouge.append(f1)

        print(f"score = {np.mean(rouge)}")
