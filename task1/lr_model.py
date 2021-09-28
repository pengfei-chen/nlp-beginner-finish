import tensorflow as tf

# TODO 目前任务是：修改TF1到TF2

class LrModel(object):
    def __init__(self, config, seq_length):
        self.config = config
        self.seq_length = seq_length
        self.w = tf.Variable(tf.random.normal(shape=(self.seq_length, config.num_classes),dtype=tf.float32))
        self.b = tf.Variable(tf.random.normal(shape=[config.num_classes],dtype=tf.float32))
        self.loss = 0.0
        self.accuracy = 0.0
        # self.lr()

    # def lr(self):
    #     self.x = tf.placeholder(tf.float32, [None, self.seq_length])
    #     """
    #     所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
    #     它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
    #     """
    #     w = tf.Variable(tf.zeros([self.seq_length, self.config.num_classes]))
    #     b = tf.Variable(tf.zeros([self.config.num_classes]))

    #     y = tf.nn.softmax(tf.matmul(self.x, w) + b)

    #     self.y_pred_cls = tf.argmax(y, 1)

    #     self.y_ = tf.placeholder(tf.float32, [None, self.config.num_classes])  # 这个才是数据输入的 y
    #     cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1]))
    #     self.loss = tf.reduce_mean(cross_entropy)

    #     self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #     # 梯度计算，参数更新
        
    #     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))  # True 或者 False
    #     self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   # tf.cast 转换 True 为 1
 
    def lr(self,x,y):
        """
        y 是数据传入的 y 
        y_  是根据w,b 计算出来的 y_ 值
        """
        # 是根据输入的 x 判断，现在的类别嘛？
        x = tf.Variable(x,dtype=tf.float32)
        y_ = tf.nn.softmax(tf.matmul(x, self.w) + self.b)
        self.y_pred_cls = tf.argmax(y_, 1)
        
        def linreg(x, w, b):
            return tf.matmul(x, w) + b
        def loss(y, y_):
            return -tf.reduce_sum(y * tf.math.log(y_))
            
        cross_entropy = tf.reduce_mean(loss(y, y_))
        self.loss = tf.reduce_mean(cross_entropy)
        
        def sgd(params, lr, grads):
            for i,param in enumerate(params):
                param.assign_sub(lr * grads[i])  #更新参数
        with tf.GradientTape() as t:
            t.watch([self.w,self.b])
            l = tf.reduce_sum(loss(y, linreg(x, self.w, self.b)))  #这一步很重要，x是变量，把它表述成关于x的表达式
        grads = t.gradient(l, [self.w,self.b])
        # 学习率是0.5时，val acc 最高47.5%：扩大学习率试下
        sgd([self.w,self.b], 1.5, grads)

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

