# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:39:45 2017
@author: DingJing
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
import requests
import io
from zipfile import ZipFile
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

'''
    获取数据：UCI 垃圾短信文本数据集(有正常短信和垃圾短信)
    用"词袋"方法处理,预测短信是否为垃圾短信
    步骤：
        1.获取数据
        2.归一化和分割数据
        3.运行词嵌入函数
        4.训练逻辑函数来预测垃圾短信
'''


''' 数据的下载 spam(垃圾短信) ham(正常短信)'''
def get_data(path):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x) >= 1]
    
    # write to csv
    with open(path, 'w') as temp_output_file:
        write = csv.writer(temp_output_file)
        write.writerows(text_data)
    return None

def normlize_data(path):
    text_data = []
    if os.path.isfile(path):
        with open(path, 'r') as temp_output_file:
            reader = csv.reader(temp_output_file)
            for row in reader:
                if len(row) > 1:
                    text_data.append(row)
    else:
        return None
    
    texts = [x[1] for x in text_data]                                                   # 内容
    target = [x[0] for x in text_data]                                                  # 结果
    
    texts = [x.lower() for x in texts]                                                  # 大写转小写
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]       # 去标点符号
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]             # 去除数字
    texts = [' '.join(x.split()) for x in texts]                                        # 去除多余空白
    
    # 正则化处理
    target = [1 if x == 'spam' else 0 for x in target]
    
    return (texts, target)

def cut_data(texts):
    text_lengths = [len(x.split()) for x in texts]
    text_lengths = [x for x in text_lengths if x < 50]
    
    plt.hist(text_lengths, bins=25)
    plt.title('histogram of words in texts')                                            # 单词长度分布直方图
    # plt.show()
    # print (len([x for x in text_lengths if x < 8 and x >= 6]))                         # 821
    sentence_size = 25
    min_word_freq = 3
    
    # 自带分词器
    vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
    vocab_processor.transform(texts)
    embedding_size = len([x for x in vocab_processor.transform(texts)])
    
    # 分割数据集为训练集和测试集
    trains_indices = np.random.choice(len(texts), round(len(texts) * 0.8), replace=False)
    tests_indices = np.array(list(set(range(len(texts))) - set(trains_indices)))

    # 训练集 和 测试集
    texts_train = [x for ix, x in enumerate(texts) if ix in trains_indices]
    texts_test = [x for ix, x in enumerate(texts) if ix in tests_indices]
    
    # 训练目标 和 测试目标
    target_train = [x for ix, x in enumerate(target) if ix in trains_indices]
    target_test = [x for ix, x in enumerate(target) if ix in tests_indices]


    # 将句子单词转成索引,再将索引转成 one-hot 向量,该向量为单位矩阵
    identity_mat = tf.diag(tf.ones(shape=[embedding_size]))
    
    A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    
    x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
    y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)
    
    x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
    x_col_sums = tf.reduce_sum(x_embed, 0)
    
    # 加入激励函数
    x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
    model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)
    
    # 声明训练模型的损失函数、预测函数和优化器
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
    prediction = tf.sigmoid(model_output)
    my_opt = tf.train.GradientDescentOptimizer(0.001)
    train_step = my_opt.minimize(loss)
    
    # 初始化计算图中的变量
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # 开始迭代训练
    loss_vec = []
    train_acc_all = []
    train_acc_avg = []
    
    for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
        y_data = [[target_train[ix]]]
        sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
        temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
        loss_vec.append(temp_loss)
        
        if(ix + 1) % 10 == 0:
            print ('Train Observation # ' + str(ix + 1) + ': Loss = ' + str(temp_loss))
                   
        [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
        train_acc_temp = target_train[ix] == np.round(temp_pred)
        train_acc_all.append(train_acc_temp)
        if len(train_acc_all) >= 50:
            train_acc_avg.append(np.mean(train_acc_all[-50:]))
      
    # Get test set accuracy
    print('Getting Test Set Accuracy For {} Sentences.'.format(len(texts_test)))
    test_acc_all = []
    for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
        y_data = [[target_test[ix]]]
        
        if (ix+1)%50==0:
            print('Test Observation #' + str(ix+1))    
        
        # Keep trailing average of past 50 observations accuracy
        # Get prediction of single observation
        [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
        # Get True/False if prediction is accurate
        test_acc_temp = target_test[ix]==np.round(temp_pred)
        test_acc_all.append(test_acc_temp)
    
    print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))
    
    # Plot training accuracy over time
    plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
    plt.title('Avg Training Acc Over Past 50 Generations')
    plt.xlabel('Generation')
    plt.ylabel('Training Accuracy')
    plt.show()
    return 

''' 主函数 '''
if __name__ == '__main__':
    local_data = "./data.csv"
    
    #sess = tf.Session()
    get_data(local_data)
    texts,target = normlize_data(local_data)
    cut_data(texts)
    exit(0)



