# python3

''' creating dot product file'''
import tensorflow as tf
from tensorflow import math
from tensorflow.keras.activations import softmax

def scaled_dot_product(q, k, v, mask):
    return q*k*v*mask