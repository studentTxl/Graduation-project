#coding=utf-8
import tensorflow as tf 
 
"""
##1.网络推理
"""
def inference(images, batch_size, n_classess):
    
    """
    第一个卷积层
    """
    # tf.variable_scope() 主要结合 tf.get_variable() 来使用，实现变量共享。下次调用不用重新产生，这样可以保存参数
    with tf.variable_scope('conv1') as scope:
         #初始化权重，[3,3,3,16]
        weights = tf.get_variable('weights', shape = [3, 3, 3, 16], dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
         #初始化偏置，16个
        biases = tf.get_variable('biases', shape=[16], dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        
        # 将偏置加在所得的值上面
        pre_activation = tf.nn.bias_add(conv, biases)
        # 将计算结果通过relu激活函数完成去线性化
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
 
    """
    池化层
    """
    with tf.variable_scope('pooling1_lrn') as scope:
        # tf.nn.max_pool实现了最大池化层的前向传播过程，参数和conv2d类似，ksize过滤器的尺寸
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='poolong1')
        # 局部响应归一化（Local Response Normalization），一般用于激活，池化后的一种提高准确度的方法。
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1, alpha=0.001/9.0, beta=0.75, name='norm1')
 
    """
    第二个卷积层
    """
    # 计算过程和第一层一样，唯一区别为命名空间
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', shape=[3,3,16,16], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    """
    第二池化层
    """
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4,bias=1,alpha=0.001/9,beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME',name='pooling2')
    
    
    """
     local3 全连接层
    """
    with tf.variable_scope('local3') as scope:
        # -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        # 获得reshape的列数，矩阵点乘要满足列数等于行数
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim,128],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[128], dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    
    """
     local4 全连接层
    """
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',shape=[128,128],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[128],dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights) + biases, name = 'local4')
    """
     lsoftmax逻辑回归
    """
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',shape=[128, n_classess],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[n_classess],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights),biases,name='softmax_linear')
        
    return softmax_linear
 
"""
##2.定义损失函数，定义传入值和标准值的差距
"""
 
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        # 计算使用了softmax回归后的交叉熵损失函数
        # logits表示神经网络的输出结果，labels表示标准答案
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='xentropy_per_example')
        # 求cross_entropy所有元素的平均值
        loss = tf.reduce_mean(cross_entropy, name='loss')
        # 对loss值进行标记汇总，一般在画loss, accuary时会用到这个函数。
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss
 
 
"""
##3.通过梯度下降法为最小化损失函数增加了相关的优化操作
"""
 
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        # 在训练过程中，先实例化一个优化函数，比如tf.train.GradientDescentOptimizer，并基于一定的学习率进行梯度优化训练
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # 设置一个用于记录全局训练步骤的单值
        global_step = tf.Variable(0, name='global_step',trainable=False)
        # 添加操作节点，用于最小化loss，并更新var_list，返回为一个优化更新后的var_list，如果global_step非None，该操作还会为global_step做自增操作
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
 
"""
##4.定义评价函数，返回准确率
"""
 
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)    # 计算预测的结果和实际结果的是否相等，返回一个bool类型的张量
        # K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。一般都是取1。
        # 转换类型
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)             #取平均值，也就是准确率
        # 对准确度进行标记汇总
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy