# -*- coding: utf-8 -*-
#"""
#    <프로그램 개요>
#    CNN Modeling 클래스
#    50x50 이미지 -> 101가지 카테고리
#    multiple instance를 허용함으로 ensemble 모델 구현 가능
#    각 Layer 마다 Tensorboard 표현을 위한 histogram 변수 저장
#    Layer 1 : 3x3, 32가지 필터, 2x2 max pooling
#    Layer 2 : 3x3, 64가지 필터, 2x2 max pooling
#    Layer 3 : 3x3, 128가지 필터, 2x2 max pooling
#    Layer 4 : 3x3, 256가지 필터, 2x2 max pooling
#    Layer 5 : Full-connected Network Layer, 4096(in)->256(out)
#    Layer 6 : Full-connected Network Layer, 256(in)->101(out)
#    
#    <Methods>
#    predict : 50x50 이미지 배열 -> 101가지 가중치 먹인 배열
#    get_accuracy : test image set를 통한 정확도 측정
#    train : train image set 학습
#"""
import tensorflow as tf

class Model:

    def __init__(self, sess, name, 
                 image_size = 50, 
                 label_size = 101,
                 learning_rate = 0.001):
        self.sess = sess
        self.name = name
        
        # image, label parameters
        self.image_size = image_size
        self.image_square = image_size * image_size
        self.label_size = label_size
        
        self.learning_rate = learning_rate
        
        self._build_net()
        self.merged_all()
            
    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, self.image_square])
            # img 50x50x1 (black/white)
            X_img = tf.reshape(self.X, [-1, self.image_size, self.image_size, 1])
            self.Y = tf.placeholder(tf.float32, [None, self.label_size])

            with tf.variable_scope('layer1'):
                # L1 ImgIn shape=(?, 50, 50, 1)
                W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
                #    Conv     -> (?, 50, 50, 32)
                #    Pool     -> (?, 25, 25, 32)
                L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
                
                tf.summary.histogram("X", self.X)
                tf.summary.histogram("weights", W1)
                tf.summary.histogram("layer1", L1)
            
            with tf.variable_scope('layer2'):
                # L2 ImgIn shape=(?, 25, 25, 32)
                W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
                #    Conv      ->(?, 25, 25, 64)
                #    Pool      ->(?, 13, 13, 64)
                L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                
                tf.summary.histogram("weights", W2)
                tf.summary.histogram("layer2", L2)
            
            with tf.variable_scope('layer3'):
                # L3 ImgIn shape=(?, 13, 13, 64)
                W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
                #    Conv      ->(?, 7, 7, 128)
                #    Pool      ->(?, 7, 7, 128)
                L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
                L3 = tf.nn.relu(L3)
                L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                                    1, 2, 2, 1], padding='SAME')
                L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
#                print(L3)
                tf.summary.histogram("weights", W3)
                tf.summary.histogram("layer3", L3)
            
            with tf.variable_scope('layer4'):
                # L4 ImgIn shape=(?, 7, 7, 128)
                W4 = tf.Variable(tf.random_normal([7, 7, 128, 256], stddev=0.01))
                #    Conv      ->(?, 7, 7, 128)
                #    Pool      ->(?, 7, 7, 128)
                #    Reshape   ->(?, 4 * 4 * 256) # Flatten them for FC
                L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
                L4 = tf.nn.relu(L4)
                L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[
                                    1, 2, 2, 1], padding='SAME')
                L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
                L4_flat = tf.reshape(L4, [-1, 256 * 4 * 4])
                
                tf.summary.histogram("weights", W4)
                tf.summary.histogram("layer4", L4)
            
            with tf.variable_scope('layer5'):
                # L5 FC 4x4x256 inputs -> 625 outputs
                W5 = tf.get_variable("W5", shape=[256 * 4 * 4, 625],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b5 = tf.Variable(tf.random_normal([625]))
                L5 = tf.nn.relu(tf.matmul(L4_flat, W5) + b5)
                L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)
                
                tf.summary.histogram("weights", W5)
                tf.summary.histogram("bias", b5)
                tf.summary.histogram("layer5", L5)
            
            with tf.variable_scope('layer6'):
                # L6 Final FC 625 inputs -> 101 outputs
                W6 = tf.get_variable("W6", shape=[625, self.label_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b6 = tf.Variable(tf.random_normal([self.label_size]))
                self.logits = tf.matmul(L5, W6) + b6
                
                tf.summary.histogram("weights", W6)
                tf.summary.histogram("bias", b6)

        with tf.variable_scope("loss"):
        # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))
            tf.summary.scalar("loss", self.cost)
            
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)
        
        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        
        with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)
            
    # Summary for tensorboard
    def merged_all(self):
        self.merged = tf.summary.merge_all()
    
    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.merged, self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})
    
if __name__ == "__main__":
    sess = tf.Session()
    m1 = Model(sess, "m1")
    sess.close()
    