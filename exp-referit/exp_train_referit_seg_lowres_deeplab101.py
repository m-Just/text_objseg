from __future__ import absolute_import, division, print_function

import argparse
import sys
import os
import tensorflow as tf
import numpy as np

from models import text_objseg_model_deeplab101 as segmodel
from util import data_reader
from util import loss

################################################################################
# Parameters
################################################################################

# Model Params
T = 20
N = 10
output_stride = 8
input_H = 320; featmap_H = (input_H // output_stride)
input_W = 320; featmap_W = (input_W // output_stride)
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Initialization Params
init_model = './exp-referit/deeplab101/checkpoints/referit_fc8_seg_lowres_init.tfmodel'

# Snapshot Params
snapshot = 5000
snapshot_folder = './exp-referit/deeplab101/checkpoints'

# Training Params
pos_loss_mult = 1.
neg_loss_mult = 1.

start_lr = 0.01
end_lr   = 0.00001
weight_decay = 0.0005
# momentum = 0.9
max_iter = 30000
is_bn_training = False

fix_convnet = 1
mlp_dropout = False
deeplab_lr_mult = 0.1

# Data Params
data_folder = './exp-referit/data/train_batch_seg_deeplab101/'
data_prefix = 'referit_train_seg'

################################################################################
# Parsed Arguments
################################################################################

parser = argparse.ArgumentParser(description="Low Resolution with Deeplab101")
parser.add_argument("--gpu", type=str, default=0,
                    help="Which gpu to use.")
parser.add_argument("--batch_size", type=int, default=N,
                    help="Number of images sent to the network in one step.")
parser.add_argument("--start_lr", type=float, default=start_lr,
                    help="Start learning rate.")
parser.add_argument("--end_lr", type=float, default=end_lr,
                    help="End learning rate.")
parser.add_argument("--max_iter", type=int, default=max_iter,
                    help="Number of training iterations.")
parser.add_argument("--fix_convnet", type=int, default=fix_convnet,
                    help="Whether keep the conv5_x layers fixed.")
parser.add_argument("--deeplab_lr_mult", type=float, default=deeplab_lr_mult,
                    help="Learning rate multiplier for finetuning deeplab network.")
parser.add_argument("--snapshot_folder", type=str, default=snapshot_folder,
                    help="Directory to save the checkpoints.")

args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if not os.path.isdir(args.snapshot_folder):
    os.mkdir(args.snapshot_folder)
snapshot_file = args.snapshot_folder + '/' + 'referit_fc8_seg_lowres_iter_%d.tfmodel'

################################################################################
# The model
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imcrop_batch = tf.placeholder(tf.float32, [N, input_H, input_W, 3])
label_batch = tf.placeholder(tf.float32, [N, featmap_H, featmap_W, 1])

# Outputs
scores = segmodel.text_objseg_full_conv(text_seq_batch, imcrop_batch,
    num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
    mlp_dropout=False, is_training=is_bn_training)

################################################################################
# Collect trainable variables, regularized variables and learning rates
################################################################################

# Only train the last layers of convnet and keep conv layers fixed
if args.fix_convnet == 1:
    train_var_list = [var for var in tf.trainable_variables()
                        if var.name.startswith('fc1_voc12') or var.name.startswith('classifier')
                        or var.name.startswith('word_embedding') or var.name.startswith('lstm')]
else: # also train the conv5_x layers
    train_var_list = [var for var in tf.trainable_variables()
                        if var.name.startswith('fc1_voc12') or var.name.startswith('classifier')
                        or var.name.startswith('word_embedding') or var.name.startswith('lstm')
                        or var.name.startswith('res5') or var.name.startswith('bn5')]
print('Collecting variables to train:')
# for var in train_var_list: print('\t%s' % var.name)
print('Done.')

# Add regularization to weight matrices (excluding bias)
reg_var_list = [var for var in tf.trainable_variables()
                if (var in train_var_list) and
                (var.name[-9:-2] == 'weights' or var.name[-8:-2] == 'Matrix')]
print('Collecting variables for regularization:')
for var in reg_var_list: print('\t%s' % var.name)
print('Done.')

# Collect learning rate for trainable variables
var_lr_mult = {var: (args.deeplab_lr_mult if var.name.startswith('res5')
                or var.name.startswith('bn5')
                else 1.0)
                for var in train_var_list}
print('Variable learning rate multiplication:')
for var in train_var_list:
    print('\t%s: %f' % (var.name, var_lr_mult[var]))
print('Done.')

################################################################################
# Loss function and accuracy
################################################################################

cls_loss = loss.weighed_logistic_loss(scores, label_batch, pos_loss_mult, neg_loss_mult)
reg_loss = loss.l2_regularization_loss(reg_var_list, weight_decay)
total_loss = cls_loss + reg_loss

################################################################################
# Solver
################################################################################

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.polynomial_decay(args.start_lr, global_step, args.max_iter,
    end_learning_rate = args.end_lr, power = 0.9)
solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
# Compute gradients
grads_and_vars = solver.compute_gradients(total_loss, var_list=train_var_list)
# Apply learning rate multiplication to gradients
grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v)
                  for g, v in grads_and_vars]
# Apply gradients
train_step = solver.apply_gradients(grads_and_vars, global_step=global_step)

################################################################################
# Initialize parameters and load data
################################################################################

snapshot_loader = tf.train.Saver(tf.trainable_variables())

# Load data
reader = data_reader.DataReader(data_folder, data_prefix)

snapshot_saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Run Initialization operations
sess.run(tf.global_variables_initializer())
snapshot_loader.restore(sess, init_model)

################################################################################
# Optimization loop
################################################################################

cls_loss_avg = 0
avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
decay = 0.99

for n_iter in range(args.max_iter):
    # Read one batch
    batch = reader.read_batch()
    text_seq_val = batch['text_seq_batch']
    imcrop_val = batch['imcrop_batch'].astype(np.float32)
    imcrop_val = imcrop_val[:,:,:,::-1] - segmodel.IMG_MEAN
    label_val = batch['label_coarse_batch'].astype(np.float32)

    # Forward and Backward pass
    scores_val, cls_loss_val, _, lr_val = sess.run([scores, cls_loss, train_step, learning_rate],
        feed_dict={
            text_seq_batch  : text_seq_val,
            imcrop_batch    : imcrop_val,
            label_batch     : label_val
        })
    cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
    print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f, lr = %f'
        % (n_iter, cls_loss_val, cls_loss_avg, lr_val))

    # Accuracy
    accuracy_all, accuracy_pos, accuracy_neg = segmodel.compute_accuracy(scores_val, label_val)
    avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
    avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
    avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg
    print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
          % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
    print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
          % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

    # Save snapshot
    if (n_iter+1) % snapshot == 0 or (n_iter+1) >= args.max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1))
        print('snapshot saved to ' + snapshot_file % (n_iter+1))

print('Optimization done.')
sess.close()
