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
#    Layer 5 : Full-connected Network Layer, 4096(in)->1024(out)
#    Layer 6 : Full-connected Network Layer, 1024(in)->101(out)
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
    
    # Define a simple convolutional layer
    def conv_layer(self, input, size_in, size_out, name="conv", keep_prob=1.0):
        with tf.variable_scope(name):
            w = tf.Variable(tf.random_normal([3, 3, size_in, size_out], stddev=0.1), name="W")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            act = tf.nn.relu(conv)
            tf.summary.histogram(name, input)
            tf.summary.histogram("weights", w)
            pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            dout = tf.nn.dropout(pool, keep_prob=keep_prob)
            return dout

    
    # And a fully connected layer
    def fc_layer(self, input, size_in, size_out, name="fc", keep_prob=1.0):
        with tf.variable_scope(name):
            w = tf.Variable(tf.random_normal([size_in, size_out], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
            act = tf.matmul(input, w) + b
            tf.summary.histogram(name, input)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            dout = tf.nn.dropout(act, keep_prob=keep_prob)
            return dout
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.x = tf.placeholder(tf.float32, [None, self.image_square])
            # img 50x50x1 (black/white)
            x_img = tf.reshape(self.x, [-1, self.image_size, self.image_size, 1])
            self.y = tf.placeholder(tf.float32, [None, self.label_size])
            
            conv1 = self.conv_layer(x_img, 1, 32, "conv1", keep_prob=self.keep_prob)
            conv2 = self.conv_layer(conv1, 32, 64, "conv2", keep_prob=self.keep_prob)
            conv3 = self.conv_layer(conv2, 64, 128, "conv3", keep_prob=self.keep_prob)
            conv4 = self.conv_layer(conv3, 128, 256, "conv4", keep_prob=self.keep_prob)
#            print(conv5)
            flattened = tf.reshape(conv4, [-1, 4 * 4 * 256])
            
            fc1 = self.fc_layer(flattened, 4 * 4 * 256, 1024, "fc1", keep_prob=self.keep_prob)
            self.logits = self.fc_layer(fc1, 1024, self.label_size, "logits", keep_prob=1.0)
            
        with tf.name_scope("cost"):
            # Compute cross entropy as our loss function
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.y))
        tf.summary.scalar("cost", self.cost)
        
        with tf.name_scope("optimizer"):
        # Use an AdamOptimizer to train the network
            self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self.cost)
        
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(
                    tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)
        
    # Summary for tensorboard
    def merged_all(self):
        self.merged = tf.summary.merge_all()
        
    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.x: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.merged, self.cost, self.optimizer], feed_dict={
            self.x: x_data, self.y: y_data, self.keep_prob: keep_prop})
            
#if __name__ == "__main__":
#    sess = tf.Session()
#    m1 = Model(sess, "m1")
#    sess.close()
    
