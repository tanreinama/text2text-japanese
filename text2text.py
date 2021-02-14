import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
import argparse
from tqdm import tqdm
from sampling import sample_sequence
from encode_bpe import BPEEncoder_ja

if int(tf.__version__[0]) > 1:
    from model import HParams as HParams
else:
    from tensorflow.contrib.training import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2ja-medium')
parser.add_argument('--output_file', type=str, default='')
parser.add_argument('--context', type=str, default='<|byte0|>')
parser.add_argument('--num_generate', type=int, default=1)
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--top_p', type=float, default=0)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--allow_duplicate_line', action='store_true')
parser.add_argument("--full_sentences", action='store_true')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

with open('ja-bpe.txt', encoding='utf-8') as f:
    bpe = f.read().split('\n')

with open('emoji.json', encoding='utf-8') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)

if os.path.isfile(args.model+'/hparams.json'):
    with open(args.model+'/hparams.json') as f:
        params = json.loads(f.read())
        hparams = HParams(**params)
        max_length = params['n_prediction']
else:
    raise ValueError('invalid model name.')

length=hparams.n_ctx - max_length - 1
temperature=args.temperature
top_k=args.top_k
top_p=args.top_p
SEP_TOKEN = enc.encode('<|byte0|>')[0]
def generate_one(sess, output):
    context_tokens = enc.encode(args.context)
    if len(context_tokens) > length:
        context_tokens = context_tokens[:length]
    context_tokens.append(SEP_TOKEN)
    out = sess.run(output, feed_dict={
        context: [context_tokens]
    })[:,len(context_tokens)-1:]
    swd = enc.decode(out[0])
    if '<|endoftext|>' in swd:
        swd = swd.split('<|endoftext|>')[0]
    if not args.allow_duplicate_line:
        swd = filter_duplicate_line(swd)
    return swd
def filter_duplicate_line(swd):
    stopword = '。｡．？！?!；;：:\r\n'
    splits = []
    spos = 0
    for i in range(0,len(swd)-1,1):
        if swd[i] in stopword:
            wd = swd[spos:i+1]
            if wd not in splits:
                splits.append(wd)
            spos = i+1
    if spos < len(swd)-1:
        wd = swd[spos:len(swd)]
        if wd not in splits:
            splits.append(wd)
    return ''.join(splits)

config = tf.ConfigProto()
if int(args.gpu) >= 0:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu
with tf.Session(config=config,graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [1, None])
    output = sample_sequence(
        hparams=hparams, length=length,
        min_length=1, context=context,
        batch_size=1,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(args.model)
    saver.restore(sess, ckpt)

    if len(args.output_file) > 0:
        with open(args.output_file, 'w', encoding='utf-8') as of:
            for i in range(args.num_generate):
                of.write(generate_one(sess, output)+'\n')
                if i < args.num_generate-1:
                    of.write('========\n')
    else:
        for i in range(args.num_generate):
            print(generate_one(sess, output))
            if i < args.num_generate-1:
                print('========')
