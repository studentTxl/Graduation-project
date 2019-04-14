#coding=utf-8
import os
import numpy as np
import tensorflow as tf
import input_data
import MainModel
 
 
from PIL import Image  
import matplotlib.pyplot as plt 
 
N_CLASSES = 2 # 二分类问题，只有是还是否，即0，1
IMG_W = 208 #图片的宽度
IMG_H = 208 #图片的高度
BATCH_SIZE = 16 #批次大小
CAPACITY = 2000  # 队列最大容量2000
MAX_STEP = 10000 #最大训练步骤
learning_rate = 0.0001  #学习率
 
"""
 定义开始训练的函数
"""
def run_training():
    
    """
    ##1.数据的处理
    """
    # 训练图片路径
    train_dir = '/home/zhang-rong/Yes/testCnn/train/'
    # 输出log的位置
    logs_train_dir = '/home/zhang-rong/Yes/testCnn/log/'
 
    # 模型输出
    train_model_dir = '/home/zhang-rong/Yes/testCnn/model/'
 
    # 获取数据中的训练图片 和 训练标签
    train, train_label = input_data.get_files(train_dir)
 
    # 获取转换的TensorFlow 张量
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
 
    """
    ##2.网络的推理
    """
    # 进行前向训练，获得回归值
    train_logits = MainModel.inference(train_batch, BATCH_SIZE, N_CLASSES)
 
    """
    ##3.定义交叉熵和 要使用的梯度下降的 优化器 
    """
    # 计算获得损失值loss
    train_loss = MainModel.losses(train_logits, train_label_batch)
    # 对损失值进行优化
    train_op = MainModel.trainning(train_loss, learning_rate)
 
    """
    ##4.定义后面要使用的变量
    """
    # 根据计算得到的损失值，计算出分类准确率
    train__acc = MainModel.evaluation(train_logits, train_label_batch)
    # 将图形、训练过程合并在一起
    summary_op = tf.summary.merge_all()
 
 
    # 新建会话
    sess = tf.Session()
  
    # 将训练日志写入到logs_train_dir的文件夹内
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()  # 保存变量
 
    # 执行训练过程，初始化变量
    sess.run(tf.global_variables_initializer())
 
 
    # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
    coord = tf.train.Coordinator()
    # 启动入队的线程，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）;
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
    """
    进行训练：
    使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，
    会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了;
    """
 
    try:
        for step in np.arange(MAX_STEP): #从0 到 2000 次 循环
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc]) 
 
 
            # 每50步打印一次损失值和准确率
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
 
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
 
 
            # 每2000步保存一次训练得到的模型
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
 
 
    # 如果读取到文件队列末尾会抛出此异常
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()       # 使用coord.request_stop()来发出终止所有线程的命令
 
    coord.join(threads)            # coord.join(threads)把线程加入主线程，等待threads结束
    sess.close()                   # 关闭会话
 
 
 
def get_one_image(data):
    '''
     获取测试数据中的，随便一张图片 ，并把它转换成数组
    '''
    n = len(data)      #训练集长度
    ind = np.random.randint(0, n)   #生成随机数
    img_dir = data[ind]    #从训练集中提取选中的图片
 
    image = Image.open(img_dir)
    plt.legend()
    plt.imshow(image)   #显示图片
    image = image.resize([208, 208])
    image = np.array(image)
    return image
 
 
 
def get_one_image_file(img_dir):
    
    image = Image.open(img_dir)
    plt.legend()
    plt.imshow(image)   #显示图片
    image = image.resize([208, 208])
    image = np.array(image)
    return image
 
 
"""
进行单张图片的测试
"""
def evaluate_one_image():
 
    # 数据集路径
    # test_dir = '/home/zhang-rong/Yes/testCnn/train/'
    # test, test_label = input_data.get_files(test_dir)
    # image_array = get_one_image(test)      #调用get_one_image随机选取一幅图片并显示
 
    image_array=get_one_image_file("/home/zhang-rong/Yes/testCnn/68.jpg")
 
    with tf.Graph().as_default():
        BATCH_SIZE = 1   # 获取一张图片
        N_CLASSES = 2  #二分类
 
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])     #inference输入数据需要是4维数据，需要对image进行resize
        logit = MainModel.inference(image, BATCH_SIZE, N_CLASSES)       
        logit = tf.nn.softmax(logit)    #inference的softmax层没有激活函数，这里增加激活函数
 
        #因为只有一副图，数据量小，所以用placeholder
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
 
        # 
        # 训练模型路径
        logs_train_dir = '/home/zhang-rong/Yes/testCnn/model/'
 
        saver = tf.train.Saver()
 
        with tf.Session() as sess:
 
            # 从指定路径下载模型
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
 
            if ckpt and ckpt.model_checkpoint_path:
                
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
 
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
 
            prediction = sess.run(logit, feed_dict={x: image_array})
            # 得到概率最大的索引
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])
                
 
"""
主函数
"""
def main():
    # run_training()
    evaluate_one_image()
 
 
if __name__ == '__main__':
	main()