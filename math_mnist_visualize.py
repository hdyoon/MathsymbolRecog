# -*- coding: utf-8 -*-
#"""
##    <프로그램 개요>
##    Main 클래스
##    CNN Model을 생성하여 Application Instance에 전달함.
##    checkpoint가 존재 한다면, Session에 CNN Model을 restore함.    
#"""
from math_application import Application
from math_mnist_cnn import Model
#from math_mnist_cnn_modulation_01 import Model

import tensorflow as tf

tf.set_random_seed(777)  # reproducibility

CHECK_POINT_DIR = TB_SUMMARY_DIR = './tb/mnist3'

tf.reset_default_graph()

# initialize
print("01.Load the tensorflow session")
sess = tf.Session()
m1 = Model(sess, "m1")
sess.close()

sess.run(tf.global_variables_initializer())

print("02.Make Check Point")
# Saver and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

print("03.Loading CNN Model")
if checkpoint and checkpoint.model_checkpoint_path:
    try:
        print(checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

# start the app
print("[INFO] starting...")

pba = Application(m1)
pba.root.mainloop()


# Test model and check accuracy
#print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))