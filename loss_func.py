# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:23:25 2018

@author: DingJing
"""
from pylab import mpl 
import tensorflow as tf
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False 

'''
    损失函数
'''
sess = tf.Session()
xVals = tf.linspace(-1., 1, 500)
target = tf.constant(0.)

# L2 正则损失函数 --- 预测值与目标值差值的平方和
l2Yvals = tf.square(target - xVals)
l2Yout = sess.run(l2Yvals)
xArray = sess.run(xVals)
plt.plot(xArray, l2Yout, 'b-', label = 'L2 正则损失函数')

# L1 正则损失  --- 预测值与目标值差的绝对值
l1Yvals = tf.abs(target - xVals)
l1Yout = sess.run(l1Yvals)
xArray = sess.run(xVals)
plt.plot(xArray, l1Yout, 'b--', label = 'L1 正则损失函数')

# Paeudo-Huber 损失函数 

plt.legend(loc='lower right', prop={'size': 11})
plt.show()

