import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import numpy as np


class RunningStats:
    def __init__(self, dim):
        self.n = 0

        self.dim = dim
        self.old_m = np.zeros(dim)
        self.new_m = np.zeros(dim)
        self.old_s = np.zeros(dim)
        self.new_s = np.zeros(dim)

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros(self.dim)
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 1.0

    def standard_deviation(self):
        return np.sqrt(self.variance()) + 1e-06

    def multiple_push(self, elements):
        for element in elements:
            self.push(element)


def var_shape(x):
    return [k.value for k in x.get_shape()]

def numel(x):
    return np.prod(var_shape(x))

def flat_grad1(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0)

def linear_search(f, x, fullstep, expected_improve_rate):
    # in numpy
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x

# symbolic vector -> symbolic tensor
def slice_vector(vector, shapes):
    assert len(vector.get_shape()) == 1
    start = 0
    tensors = []
    for shape in shapes:
        size = np.prod(shape)
        tensor = tf.reshape(vector[start:(start + size)], shape)
        tensors.append(tensor)
        start += size
    return tensors

# conjugate gradient to solve Ax=b
from numpy.linalg import inv
def conjugate_gradient(f_Ax, b, cg_iter=10, residual_tolerance=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iter):
        z = f_Ax(p)
        v = rdotr / (p.dot(z) + 1e-3)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tolerance:
            break
    return x

def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)

def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)

    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5)
    return kl
