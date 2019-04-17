#coding=utf-8
import tensorflow as tf 
import numpy as np 
import os 
 
 
"""
训练图片  路径    猫是没有进行复制粘贴篡改的图片，狗是进行过复制粘贴篡改的照片，先进行模型的训练之后进行内容的修改
"""
train_dir="/c/Graduation-project/Tamper_Detection_1/train"
 
 
"""
 获取数据，并处理 数据 
"""
def get_files(file_dir):
    cats=[] #未进行篡改图片 列表
    lable_cats=[] #未进行篡改图片的标签 列表
    dogs=[] #的图片 列表
    lable_dogs=[]  #狗的标签 列表
 
    #os.listdir为列出路径内的所有文件
    for file in os.listdir(file_dir):
        name = file.split('_')   #将每一个文件名都进行分割，以.分割
        #这样文件名 就变成了三部分 name的形式 [‘dog’，‘9981’，‘jpg’]
        if name[0]=='Au':
            cats.append(file_dir+"/"+file)   #在定义的cats列表内添加图片路径，由文件夹的路径+文件名组成
            lable_cats.append(0) #在猫的标签列表中添加对应图片的标签，猫的标签为0，狗为1
        else:
            dogs.append(file_dir+"/"+file)
            lable_dogs.append(1)
    print(" %d cat, %d dog"%(len(cats),len(dogs)))
    image_list = np.hstack((cats, dogs))  #将猫和狗的列表合并为一个列表
    label_list = np.hstack((lable_cats, lable_dogs)) #将猫和狗的标签列表合并为一个列表
 
    #将两个列表构成一个数组
    temp=np.array([image_list,label_list])
    temp=temp.transpose() #将数组矩阵转置
    np.random.shuffle(temp) #将数据打乱顺序，不再按照前边全是猫，后面全是狗这样排列
 
    image_list=list(temp[:,0]) #图片列表为temp 数组的第一个元素
    label_list = list(temp[:, 1]) #标签列表为temp数组的第二个元素
    label_list = [int(i) for i in label_list] #转换为int类型
    #返回读取结果，存放在image_list,和label_list中
    return image_list, label_list
 
 
"""
将图片转为 tensorFlow 能读取的张量
"""
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #数据转换
    image = tf.cast(image, tf.string)   #将image数据转换为string类型
    label = tf.cast(label, tf.int32)    #将label数据转换为int类型
    #入队列
    input_queue = tf.train.slice_input_producer([image, label])
    #取队列标签 张量
    label = input_queue[1] 
    #取队列图片 张量
    image_contents = tf.read_file(input_queue[0])
 
    #解码图像，解码为一个张量
    image = tf.image.decode_jpeg(image_contents, channels=3)
 
    #对图像的大小进行调整，调整大小为image_W,image_H
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #对图像进行标准化
    image = tf.image.per_image_standardization(image)
 
    #等待出队
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
 
    label_batch = tf.reshape(label_batch, [batch_size]) #将label_batch转换格式为[]
    image_batch = tf.cast(image_batch, tf.float32)   #将图像格式转换为float32类型
  
    return image_batch, label_batch  #返回所处理得到的图像batch和标签batch