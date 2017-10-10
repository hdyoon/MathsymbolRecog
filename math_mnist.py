# -*- coding: utf-8 -*-
#"""
#    <프로그램 개요>
#    DataSet Class
#    Train, Test, Validation을 객체화하여 손쉽게 훈련/검증을 할 수 있도록 하는 DataSet 객체
#    MNIST와 유사하게 동작하게 하여 다른 샘플 코드 변경을 최소화한다.
#    
#    <Methods>
#    __init__ : 멤버 변수 초기화. 
#                fake_data, one_hot은 무시, 아직 미완성
#    next_batch : batch_size 만큼 Dataset에서 Flush
#    read_categories : categories.txt에서 공백을 기준으로 split 후 1hot 형태의
#                      pandas.DataFrame으로 반환
#    get_label1hot : Symbol Character -> 1hot 변환
#    get_category_char : index 번호 또는 1hot -> Symbol Character
#    extract_datas : pickle 객체로 부터 image data와 label data로 분리 추출, 같은 인덱스 위치
#    read_data_sets : extract_datas를 통해 train, test, validation 3개의 DataSet으로 추출 후
#                     tensorflow.contrib.learn.python.learn.datasets내의 3개 멤버객체 변수(train,test,validation)에 저장
#"""

import numpy
import os
import _pickle as pickle
import csv
from six.moves import xrange  # pylint: disable=redefined-builtin

#sklearn.preprocessing.OneHotEncoder

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

#import random
#import matplotlib.pyplot as plt

image_pixels_size = 50
num_of_categories = 101
categories_file_name = 'categories.txt'

class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=True,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
            
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

# Convert shape from [num examples, rows, columns, depth]
# to [num examples, rows*columns] (assuming depth == 1)
#        if reshape:
#            assert images.shape[3] == 1
#            images = images.reshape(images.shape[0],
#                                    images.shape[1] * images.shape[2])
        if dtype == dtypes.float32:
# Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 1.0)
#            images = numpy.multiply(images, 1.0 / 255.0)
        
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        pass
    
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * image_pixels_size**2
            if self.one_hot:
                fake_label = [1] + [0] * num_of_categories
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                    fake_label for _ in xrange(batch_size)
                    ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def get_categories():
    i = 0
    cgs = []
    reader = csv.reader(open(categories_file_name), delimiter=' ')
    for row in reader:
        for s in row:
            cgs.append(s)
            i += 1
    assert len(cgs) == len(list(set(cgs))), "There is duplicate category value exist."
    return numpy.array(cgs)

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def one_hot_to_dense(labels_one_hot):
    assert labels_one_hot.ndim==2, "numpy array must be 2-D array."
    labels_dense = numpy.argmax(labels_one_hot,axis=1)
    return labels_dense

def extract_datas(f, labels_one_hot=True):
    data_list = pickle.load(f)
    images_array = numpy.asarray(list(map(lambda x: x['pattern'].reshape((image_pixels_size**2)), data_list)))
    categories = get_categories()
    labels_array = numpy.asarray(list(map(lambda x: numpy.where(x['label']==categories), data_list))).flatten()
    if labels_one_hot==True:
        labels_array = dense_to_one_hot(labels_array,101)
    return images_array, labels_array

def read_data_sets(outputs_rel_path='outputs',
                   fake_data=False,
                   one_hot=True,
                   dtype=dtypes.float32,
                   validation_size=100):
    if fake_data:

        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)
    
    train_dir = os.path.join(outputs_rel_path, 'train')
    test_dir = os.path.join(outputs_rel_path, 'test')
#    validation_dir = os.path.join(outputs_rel_path, 'validation')
    'Load pickled data'
    with open(os.path.join(train_dir, 'train.pickle'), 'rb') as train_f:
        print('Restoring training set ...')
        train_images,train_labels=extract_datas(train_f,labels_one_hot=one_hot)
    with open(os.path.join(test_dir, 'test.pickle'), 'rb') as test_f:
        print('Restoring test set ...')
        test_images,test_labels=extract_datas(test_f,labels_one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
      
    train = DataSet(train_images,train_labels)
    test = DataSet(test_images,test_labels)
    validation = DataSet(validation_images, validation_labels)
    return  base.Datasets(train=train, validation=validation, test=test)

if __name__ == "__main__":
    
    cgs = get_categories()
    print(cgs)
    mnist = read_data_sets()
    batch_images, batch_labels = mnist.validation.next_batch(100)
    print(batch_images)
    print(batch_labels)
    print(one_hot_to_dense(batch_labels))
    