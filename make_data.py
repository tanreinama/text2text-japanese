import numpy as np
import jaconv
import re
from copy import copy

from encode_bpe import BPEEncoder_ja

def make_rouge(raw_tokens):
    global split_types
    sentences = []
    if len(raw_tokens) < 2:
        return None
    for sp in split_types:
        if sp in raw_tokens:
            rt = copy(raw_tokens)
            while sp in rt:
                ind = rt.index(sp)
                tok = rt[:ind+1]
                if len(tok) > 0:
                    sentences.append(tok)
                rt = rt[ind+1:]
            break
    if len(sentences) < 2:
        return None
    maxscore = -1
    maxothers = []
    maxindex = 0
    for i in range(len(sentences)):
        others = [copy(sentences[j]) for j in range(len(sentences)) if j!=i]
        others = sum(others, [])
        if len(sentences[i]) > 0 and len(others) > 0:
            inter = set(sentences[i]) & set(others)
            ninter = len(inter)
            recall = ninter / len(sentences[i])
            precision = ninter / len(others)
            if recall+precision > maxscore:
                maxscore = recall+precision
                maxothers = others
                maxindex = i
    return maxothers, sentences[maxindex]

if __name__=='__main__':
    import argparse
    import os
    import json
    from tqdm import tqdm
    import pickle
    from multiprocessing import Pool
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="source dir", required=True )
    parser.add_argument("--dst_file", help="destnation file", required=True )
    parser.add_argument("--num_process", help="process num", type=int, default=8 )
    parser.add_argument("--split_tag", help="text split tag", default='' )
    args = parser.parse_args()

    with open('ja-bpe.txt', encoding='utf-8') as f:
        bpe = f.read().split('\n')
    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())
    enc = BPEEncoder_ja(bpe, emoji)
    split_types = enc.encode('。．｡、；：.,「『')
    eot_token = enc.encode('<|endoftext|>')

    array_file = []
    def _proc(i):
        token_chunks = []
        for j, input in enumerate(tqdm(array_file)):
            if not ((j % args.num_process) == i):
                continue
            with open(input, 'r', encoding='utf-8') as fp:
                raw_text = fp.read()
                if args.split_tag == '':
                    tokens = make_rouge(enc.encode(raw_text))
                else:
                    tk = raw_text.split(args.split_tag)
                    tokens = [enc.encode(tk[0]),enc.encode(tk[1])]
                if tokens is not None and len(tokens[0]) > 0 and len(tokens[1]) > 0:
                    token_chunks.append(tokens)
        with open(f'{args.dst_file}_{i}.pkl', 'wb') as f:
            pickle.dump(token_chunks, f)

    for curDir, dirs, files in os.walk(args.src_dir):
        print('append #',curDir)
        for file in tqdm(files):
            if file.endswith(".txt"):
                input = os.path.join(curDir, file)
                array_file.append(input)

    with Pool(args.num_process) as p:
        p.map(_proc, list(range(args.num_process)))
