########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, toimage, imsave
from imagenet_classes import class_names
import double2image

def GradCam( probs, visualization_layer, category = None):
    with tf.name_scope('GradCam') as scope:
        with tf.name_scope('Extract_label') as scope:
            if category == None:
                # if category is unspecificed, use the class with highest prob
                category = tf.argmax(probs[0], output_type=tf.int32) 
            target = probs[0, category]
        with tf.name_scope('Get_gradients') as scope:
            grads = tf.gradients(target,
                                 visualization_layer,
                                 grad_ys=None,
                                 name='gradients')[0]
        with tf.name_scope('get_weights') as scope:
            weights = tf.reduce_sum(grads, axis = [0, 1, 2])
        with tf.name_scope('weighted_sum') as scope:
            heatmap = tf.squeeze(tf.tensordot(visualization_layer, weights, [[3], [0]]))
        return tf.nn.relu(heatmap)    
def Guided_backprop(imgs, probs, category = None):
        with tf.name_scope('Extract_labels') as scope:
            if category == None:
                # if category is unspecificed, use the class with highest prob
                category = tf.argmax(probs[0], output_type=tf.int32) 
            target = probs[0, category]
            labels = np.array([1 if i == category else 0 for i in range(probs.shape[1])])
        with tf.name_scope('deconvolution') as scope:
            cost = tf.reduce_sum((probs - labels) ** 2)
            grads = tf.gradients(cost,
                                 imgs,
                                 grad_ys=None,
                                 name='gradients')
        return tf.squeeze(grads)

        
