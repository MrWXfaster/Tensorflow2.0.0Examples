
#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : XH
#   File name   : config.py
#   Author      : WrWx
#   Created date: 2019-07-20 09:06:45
#   Description :
#
#================================================================
from __future__ import absolute_import,division,print_function
import tensorflow as tf
'''
1.Tensor
'''
print(tf.add(1,2))
print(tf.add([3,8],[2,5]))
print(tf.square(6))
print(tf.reduce_sum([7,8,9]))
print(tf.square(3) + tf.square(4))
x = tf.matmul([[3],[6]],[[2]])
print(x)
print(x.shape)
print(x.dtype)

import numpy as np
ndarray = np.ones([2,2])
tensor = tf.multiply(ndarray,36)
print(tensor)
#使用np.add对tensorflow进行加速
print(np.add(tensor,1))
#转化为numpy类型
print(tensor.numpy())

'''
2.GPU加速
'''

x = tf.random.uniform([3,3])
print("Is GPU availabel:")
print(tf.test.is_gpu_available())
print("Is the tensor on gpu #0:")
print(x.device.endswith("GPU:0"))

#显示设备放置（Placement）f.device上下文管理器将TensorFlow操作显式放置在特定设备上，
import time
def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x,x)
    result = time.time() - start
    print('10 loops: {:0.2}ms'.format(1000*result))
'''
新增了一种格式化字符串的函数 str.format()，它增强了字符串格式化的功能。

基本语法是通过 {} 和 : 来代替以前的 % 。

format 函数可以接受不限个参数，位置可以不按顺序，0.2表示小数点后保留的个数
'''

#强制使用cpu
print("On CPU")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000,1000])
    # 使用断言验证当前是否为CPU0
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# 如果存在GPU，强制使用GPU
if tf.test.is_gpu_available():
    print("On GPU:")
    with tf.device.endswith("GPU:0"):
        x = tf.random.uniform([1000,1000])
    #使用断言验证当前是否为GPU0
    assert x.device.endswith("GPU:0")
    time_matmul(x)

'''
3.数据集
'''
#从列表中获取tensor
ds_tensors = tf.data.Dataset.from_tensor_slices([6,5,4,3,2,1])
#创建csv文件

import tempfile
_, filename = tempfile.mkstemp()
print(filename)

with open(filename, "w") as f:
    f.write("""line1,line2,line3""")
#获取textLineDataset数据集
ds_file = tf.data.TextLineDataset(filename)

#使用map，batch和shuffle等转换函数将转换应用于数据集记录。
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print("ds_tensors中的元素：")
for x in ds_tensors:
    print(x)
#从文件中读取的对象创建的数据源
print("\ndsfile中的元素：")
for x in ds_file:
    print(x)