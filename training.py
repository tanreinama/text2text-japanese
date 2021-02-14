import argparse
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
import time
import tqdm
import glob
import pickle
from copy import copy

from encode_bpe import BPEEncoder_ja
import model

if int(tf.__version__[0]) > 1:
    from model import HParams as HParams
else:
    from tensorflow.contrib.training import HParams

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

parser = argparse.ArgumentParser(
    description='Pretraining TEXT2TEXT-JA on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input pkl file')
parser.add_argument('--base_model', type=str, default='gpt2ja-medium', help='a path to a model file')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--optim', type=str, default='adam', help='"adam", "adagrad", or "sgd" to use optimizer')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=3e-6, help='Learning rate for optimizer')
parser.add_argument('--warmup_steps', metavar='WR', type=int, default=0, help='Learning rate warming up steps')
parser.add_argument('--max_train_steps', default=-1, type=int, help='Maximum training steps')

parser.add_argument('--run_name', type=str, default='text2text_ja-medium', help='Run id. Name of subdirectory in checkpoint/')
parser.add_argument('--save_every', metavar='N', type=int, default=10000, help='Write a checkpoint every N steps')
parser.add_argument('--max_answer_len', default=160, type=int, help='Maximum number of answer tokens')

parser.add_argument('--gpu', default='0', help='visible gpu number.')
parser.add_argument('--train_type', default='QtoA', help='Direction to generate text.')

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

with open('ja-bpe.txt', encoding='utf-8') as f:
    bpe = f.read().split('\n')

with open('emoji.json', encoding='utf-8') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)
eot_token = enc.encode('<|endoftext|>')[0]
sep_token = enc.encode('<|byte0|>')[0]

def get_masked_lm_output(hparams, logits, positions, label_ids, label_weights):
    logits = gather_indexes(logits, positions)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
            label_ids, depth=n_vocab, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    return (loss, per_example_loss, log_probs)

def gather_indexes(sequence_tensor, positions):
    sequence_shape = model.shape_list(sequence_tensor)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                                                        [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor

def main():
    args = parser.parse_args()

    if 'small' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 768,
          "n_head": 12,
          "n_layer": 12
        })
    elif 'medium' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 1024,
          "n_head": 16,
          "n_layer": 24
        })
    elif 'large' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 1280,
          "n_head": 20,
          "n_layer": 36
        })
    else:
        raise ValueError('invalid model name.')

    max_answer_len = args.max_answer_len
    batch_size = args.batch_size
    max_seq_length = hparams.n_ctx

    if args.train_type == 'QtoA':
        index_q = 0
        index_a = 1
        max_q = max_seq_length - args.max_answer_len
        max_a = args.max_answer_len
    elif args.train_type == 'AtoQ':
        index_q = 1
        index_a = 0
        max_q = args.max_answer_len
        max_a = max_seq_length - args.max_answer_len
    else:
        raise ValueError('invalid train type.')

    config = tf.ConfigProto()
    if int(args.gpu) >= 0:
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = args.gpu
    with tf.Session(config=config) as sess:
        input_ids = tf.placeholder(tf.int32, [batch_size, None])
        masked_lm_positions = tf.placeholder(tf.int32, [batch_size, None])
        masked_lm_ids = tf.placeholder(tf.int32, [batch_size, None])
        masked_lm_weights = tf.placeholder(tf.float32, [batch_size, None])

        output = model.model(hparams=hparams, X=input_ids, past=None, reuse=tf.AUTO_REUSE)

        (loss,_,_) = get_masked_lm_output(hparams=hparams, logits=output['logits'],
                        positions=masked_lm_positions, label_ids=masked_lm_ids, label_weights=masked_lm_weights)

        train_vars = tf.trainable_variables()

        global_step = tf.Variable(0, trainable=False)
        if args.warmup_steps > 0:
            learning_rate = tf.compat.v1.train.polynomial_decay(
                    learning_rate=1e-10,
                    end_learning_rate=args.learning_rate,
                    global_step=global_step,
                    decay_steps=args.warmup_steps
                )
        else:
            learning_rate = args.learning_rate

        if args.optim=='adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.99,
                                           epsilon=1e-7)
        elif args.optim=='adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif args.optim=='sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError('invalid optimizer name.')

        train_vars = tf.trainable_variables()
        opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)

        summaries = tf.summary.scalar('loss', loss)
        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.latest_checkpoint(args.base_model)
        saver.restore(sess, ckpt)
        print('Loading checkpoint', ckpt)

        print('Loading dataset...')
        global_chunks = []
        for fn in glob.glob(args.dataset):
            with open(fn, 'rb') as f:
                for p in pickle.load(f):
                    if len(p[0]) > 0 and len([1]) > 0:
                        if p[0][-1] != eot_token:
                            p[0].append(eot_token)
                        if p[1][-1] != eot_token:
                            p[1].append(eot_token)
                        global_chunks.append(p)
        global_chunk_index = np.random.permutation(len(global_chunks))
        global_chunk_step = 0
        print('There is',len(global_chunks),'chunks.')
        print('Training...')

        def sample_feature():
            nonlocal global_chunks,global_chunk_index,global_chunk_step

            p_input_ids = []
            p_masked_lm_positions = []
            p_masked_lm_ids = []
            p_masked_lm_weights = []

            for b in range(batch_size):
                idx = global_chunk_index[global_chunk_step]
                global_chunk_step += 1
                if global_chunk_step >= len(global_chunk_index):
                    global_chunk_step = 0
                    global_chunk_index = np.random.permutation(len(global_chunks))
                sampled_tokens = global_chunks[idx]

                # Make Sequence
                ids = copy(sampled_tokens[index_q])
                if len(ids) > max_q:
                    ids = ids[:max_q]
                ids[-1] = sep_token

                lm_ids = copy(sampled_tokens[index_a])
                if len(lm_ids) > max_a:
                    lm_ids = lm_ids[:max_a]
                lm_weights = [1.0] * len(lm_ids)
                lm_positions = list(range(len(ids)-1,len(ids)-1+len(lm_ids),1))
                while len(lm_positions) < max_answer_len:
                    lm_positions.append(0)
                    lm_ids.append(0)
                    lm_weights.append(0.0)

                ids = ids + lm_ids
                while len(ids) < max_seq_length:
                    ids.append(eot_token)

                p_input_ids.append(ids)
                p_masked_lm_positions.append(lm_positions)
                p_masked_lm_ids.append(lm_ids)
                p_masked_lm_weights.append(lm_weights)

            return {input_ids:p_input_ids,
                    masked_lm_positions:p_masked_lm_positions,
                    masked_lm_ids:p_masked_lm_ids,
                    masked_lm_weights:p_masked_lm_weights}

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        hparams_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'hparams.json')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')
            with open(hparams_path, 'w') as fp:
                fp.write(json.dumps({
                      "n_vocab": int(hparams.n_vocab),
                      "n_ctx": int(hparams.n_ctx),
                      "n_embd": int(hparams.n_embd),
                      "n_head": int(hparams.n_head),
                      "n_layer": int(hparams.n_layer),
                      "n_prediction": int(max_answer_len),
                }))

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:
                if counter % args.save_every == 0:
                    save()

                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summaries),
                    feed_dict=sample_feature())

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter = counter+1
                if args.warmup_steps > 0:
                    global_step = global_step+1
                if args.max_train_steps > 0 and args.max_train_steps <= counter:
                    save()
                    break
        except KeyboardInterrupt:
            print('interrupted')
            save()


if __name__ == '__main__':
    main()
