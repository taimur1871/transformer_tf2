# python3

''' creating dot product file'''
import tensorflow as tf
from tensorflow import math
from tensorflow.keras.activations import softmax
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk using dk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk/math.sqrt(dk)

    # add mask to scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask *-1e9)

    # softmax is normalized on last axis (seq_len_k) so that 
    # the socres add up to 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)