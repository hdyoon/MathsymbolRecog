# -*- coding: utf-8 -*-
#"""
#    <Script Info.>
#    Train Only
#    TRAINING_EPOCHS 만큼 모델 훈련 후
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
import embedder

import math_mnist
from math_mnist_cnn import Model
# import matplotlib.pyplot as plt

def main():

    # get a number of epochs by argument
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", 
                        help="[integer]number of training epochs", 
                        type=int)
    args = parser.parse_args()

    # hyper parameters
    TRAINING_EPOCHS = args.epochs
    BATCH_SIZE = 100
    
    # image, label parameters for embedder
    IMAGE_SIZE = 50     # one side pixel size of the square
    LABEL_SIZE = 101    # number of labels(categories)
    VALIDATION_SIZE = 1024  # validation data size for embedded visualization
    NUM_CHANNELS = 1    # image channel (GREY, RGB, RGBA...)
    SPRITES = 'sprite.png'
    LABELS = 'labels.tsv'
    
    # directory for session checkpoint and tensorboard Saver
    CHECK_POINT_DIR = TB_SUMMARY_DIR = '.\\tb\\mnist2'
       
    # initialize for tensorflow
    tf.set_random_seed(777)  # reproducibility
    tf.reset_default_graph()    # reset tf graph
    last_epoch = tf.Variable(0, name='last_epoch')  #save the last epochs
    
    # load handwritten math symbol datas to dataset obj.
    mnist = math_mnist.read_data_sets(validation_size=VALIDATION_SIZE)

    # load the model(math_mnist_cnn)
    sess = tf.Session()
    m1 = Model(sess, "m1")
    sess.close()
    
    sess.run(tf.global_variables_initializer())
    
    # Create summary writer
    writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)    #add graph to summary writer
#    global_step = 0

    # make embedded data for validation
    validation_images = mnist.validation.images
    embedder.make_sprite(mnist.validation.images,
                         IMAGE_SIZE,NUM_CHANNELS,CHECK_POINT_DIR)
    
    # make embedded label for validation
    metadata_file = open(os.path.join(TB_SUMMARY_DIR, LABELS), 'w')
    df_labels = math_mnist.read_categories()
    for label in mnist.validation.labels:
        metadata_file.write('%s\n' % math_mnist.get_category_char(label, df_labels))
    metadata_file.close()
    
    embedding = tf.Variable(tf.zeros([VALIDATION_SIZE, LABEL_SIZE]), name="test_embedding")
    assignment = embedding.assign(m1.get_logits)
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = SPRITES
    embedding_config.metadata_path = LABELS
    
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([IMAGE_SIZE, IMAGE_SIZE])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    
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
    for epoch in range(start_from, TRAINING_EPOCHS):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / BATCH_SIZE)
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
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
        
        sess.run(last_epoch.assign(epoch + 1))
        
        if epoch % 5 == 0:
            
            sess.run(assignment, feed_dict={m1.x:validation_images, m1.keep_prob: 1.0})
        
            if not os.path.exists(CHECK_POINT_DIR):
                os.makedirs(CHECK_POINT_DIR)
#            #Fix Unicode Error
#            CHECK_POINT_DIR = CHECK_POINT_DIR.encode('utf-8', 'surrogateescape').decode('ISO-8859-1')
            print("Saving network...")
            saver.save(sess, CHECK_POINT_DIR + "\\model", global_step=i)
        
    print('Learning Finished!')

if __name__ == "__main__":
    main()


#saver.save(sess, CHECK_POINT_DIR + "\\model", global_step=856)


#print(activations)
#print(m1.get_logits)
#print(mnist.validation.labels)
# summary embedding
#embedder.summary_embedding(sess=sess, dataset=mnist.validation.images, embedding_list=[activations],
#                           embedding_path=TB_SUMMARY_DIR,
#                           image_size=IMAGE_SIZE, channel=NUM_CHANNELS, labels=mnist.validation.labels)

## Test model and check accuracy
#print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
