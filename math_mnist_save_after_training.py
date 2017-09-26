# -*- coding: utf-8 -*-
#"""
#    <Script Info.>
#    Train Only
#    training_epochs 만큼 모델 훈련 후
#    정해진 directory(CHECK_POINT_DIR)에 현재 모델 상태 저장
#    마지막에 정확도 테스트 리포팅    
#    
#    Usage : math_mnist_save_after_training.py 100
#    Spyder의 경우 Run > Configuration per file > Command line options 에 epochs횟수(정수) 추가
#"""
import tensorflow as tf
import os
import time
import argparse

import math_mnist
from math_mnist_cnn import Model

# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("epochs", 
                    help="[integer]number of training epochs", 
                    type=int)
args = parser.parse_args()

tf.set_random_seed(777)  # reproducibility
mnist = math_mnist.read_data_sets() # load math

# hyper parameters
learning_rate = 0.001
training_epochs = args.epochs    #a input argument
batch_size = 100

# image, label parameters
image_size = 50     # one side pixel size of the square
image_square = image_size * image_size
label_size = 101    # number of labels(categories)

CHECK_POINT_DIR = TB_SUMMARY_DIR = '.\\tb\\mnist2'

tf.reset_default_graph()

last_epoch = tf.Variable(0, name='last_epoch')

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.close()

sess.run(tf.global_variables_initializer())

# Create summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)
global_step = 0

# Saver and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

start_from = sess.run(last_epoch)

# train my model
print('Start learning from:', start_from)

# train my model
for epoch in range(start_from, training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch
        writer.add_summary(summary, i)
#        print("epoch: ",epoch," i: ",i)
        
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print(
            'Epoch:', '%04d' % (epoch + 1), 
            'cost =', '{:.9f}'.format(avg_cost),
            'time =', s
    )
    print("Learning Rate : ", m1.get_accuracy(mnist.test.images, mnist.test.labels))
        
    print("Saving network...")
    
    sess.run(last_epoch.assign(epoch + 1))
    
    if not os.path.exists(CHECK_POINT_DIR):
        os.makedirs(CHECK_POINT_DIR)
    
    #Fix Unicode Error
    CHECK_POINT_DIR = CHECK_POINT_DIR.encode('utf-8', 'surrogateescape').decode('ISO-8859-1')
    saver.save(sess, CHECK_POINT_DIR + "\\model", global_step=i)
    
print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
