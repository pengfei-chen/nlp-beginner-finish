import tensorflow as tf

# TODO 目前任务是：修改TF1到TF2

class LrModel(object):
    def __init__(self, config, seq_length, x, y, batch_size=1):
        self.config = config
        self.seq_length = seq_length
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.lr()

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
 
    def lr(self):
        # 是根据输入的 x 判断，现在的类别嘛？
        self.x = tf.Variable(self.x,dtype=tf.float32)
        y = tf.nn.softmax(tf.matmul(self.x, w) + b)
        self.y_pred_cls = tf.argmax(y, 1)
        
        def linreg(x, w, b):
            return tf.matmul(x, w) + b
        def loss(y, y_):
            return -tf.reduce_sum(y * tf.math.log(y_))
            
        cross_entropy = tf.reduce_mean(loss(self.y, y))
        self.loss = tf.reduce_mean(cross_entropy)
        
        def sgd(params, lr, batch_size, grads):
            for i,param in enumerate(params):
                param.assign_sub(lr * grads[i]/batch_size)
        with tf.GradientTape() as t:
            t.watch([w,b])
            l = tf.reduce_sum(loss(self.y, linreg(self.x, w, b)))
        grads = t.gradient(l, [w,b])
        sgd([w,b], 0.5, self.batch_size, grads)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

