#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : XH
#   File name   : config.py
#   Author      : WrWx
#   Created date: 2019-07-20 09:06:45
#   Description : 基础MLP网络
#
#================================================================
'''
1.回归任务
'''
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
print(tf.__version__)
#导入数据
(x_train,y_train),(x_test,y_test) = keras.datasets.boston_housing.load_data()
print(x_train.shape,"",y_train.shape)
print(x_test.shape,"",y_test.shape)
#构建模型
model = keras.Sequential([layers.Dense(32,activation="sigmoid",input_shape=(13,)),
                          layers.Dense(32,activation="sigmoid"),
                          layers.Dense(32,activation="sigmoid"),
                          layers.Dense(1)
                          ])
#配置模型
model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss="mean_squared_error",
              metrics=["mse"])
model.summary()
#训练
model.fit(x_train,y_train,batch_size=50,epochs=50,validation_split=0.1,verbose=1)
result = model.evaluate(x_test,y_test)

print(model.metrics_names)
print(result)

'''
2.分类任务 
'''

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#导入数据
whole_data = load_breast_cancer()
x_data = whole_data.data
y_data = whole_data.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=7)

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

#构建模型
#构建模型
model = keras.Sequential([layers.Dense(32,activation="relu",input_shape=(30,)),
                          layers.Dense(32,activation="relu"),
                          layers.Dense(1,activation="sigmoid"),
                          ])
#配置模型
model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=["accuracy"])
model.summary()

#训练
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
#模型评估
model.evaluate(x_test,y_test)
print(model.metrics_names)