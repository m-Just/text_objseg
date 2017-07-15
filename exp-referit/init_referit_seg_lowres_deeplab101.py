from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

import tensorflow as tf

from models import text_objseg_model_deeplab101 as segmodel
from deeplab_resnet import model as deeplab101
from six.moves import cPickle

################################################################################
# Parameters
################################################################################

seg_model = './exp-referit/deeplab101/checkpoints/referit_fc8_seg_lowres_init.tfmodel'
pretrained_params = './tensorflow-deeplab-resnet/models/deeplab_resnet_init.ckpt'

# Model Params
T = 20
N = 1

num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500
is_bn_training = False

################################################################################
# pretrained deeplab101 model
################################################################################

# Inputs
imcrop_batch = tf.placeholder(tf.float32, [N, 320, 320, 3])

print('Loading deeplab101 weights')
# Load pretrained deeplab101 model and fetch weights
net = deeplab101.DeepLabResNetModel({'data': imcrop_batch}, is_training=is_bn_training, num_classes=embed_dim)
# We need to randomly initialize the last layer
restored_var = [var for var in tf.global_variables() if 'fc1_voc12' not in var.name]

snapshot_loader = tf.train.Saver(var_list=restored_var)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  snapshot_loader.restore(sess, pretrained_params)
  variable_dict = {var.name:var.eval(session=sess) for var in tf.global_variables()}
print("Done")

# Clear the graph
tf.reset_default_graph()


################################################################################
# low resolution segmentation network
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])  # one batch per sentence
imcrop_batch = tf.placeholder(tf.float32, [N, 320, 320, 3])

print('Saving deeplab101 segmodel weights')

_ = segmodel.text_objseg_full_conv(text_seq_batch, imcrop_batch,
  num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
  mlp_dropout=False, is_training=is_bn_training)

# Assign outputs
assign_ops = []
for var in tf.global_variables():
  if var.name not in variable_dict:
    print('not in dict: ' + var.name)
    continue
  assign_ops.append(tf.assign(var, variable_dict[var.name].reshape(var.get_shape().as_list())))

# Save segmentation model initialization
snapshot_saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.group(*assign_ops))
  snapshot_saver.save(sess, seg_model)
