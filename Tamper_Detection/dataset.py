import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
 
#分别读取，处理数据
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []
 
    print('Going to read training images')
	#大类的循环
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*.jpg')
		#把当前路径下所有数据取到手
        files = glob.glob(path)
		#每个类别自己的循环
        for fl in files:
            image = cv2.imread(fl)
			#将图片转换为64*64
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
			#像素点归一化
            image = np.multiply(image, 1.0 / 255.0)
			#处理好的图片放入images中
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
			#每张图片对应的（标签）
            labels.append(label)
            flbase = os.path.basename(fl)
			#图片的名字
            img_names.append(flbase)
			#图像的类别
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
 
    return images, labels, img_names, cls
 
 
class DataSet(object):
 
  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]
 
    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0
 
  @property
  def images(self):
    return self._images
 
  @property
  def labels(self):
    return self._labels
 
  @property
  def img_names(self):
    return self._img_names
 
  @property
  def cls(self):
    return self._cls
 
  @property
  def num_examples(self):
    return self._num_examples
 
  @property
  def epochs_done(self):
    return self._epochs_done
 
#取数据	
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
 
    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
 
    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]
 
#返回分类好的训练集和验证集
def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()
 
  #读取图像
  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  #将图片打乱
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  
 
  #判断一个对象是否是一个已知的类型 如果参数1与参数2的类型相同则返回true
  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])		#shape[0] 总体的个数
 
  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]
 
  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]
 
  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
 
  return data_sets