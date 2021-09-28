import time
from datetime import timedelta
from datahelper.data_process import DataProcess
from config.lr_config import LrConfig
from lr_model import LrModel
import tensorflow as tf


def get_time_dif(start_time):
    """获取已经使用的时间"""
    end_time = time.time()
    time_dif = end_time-start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(x_, y_ ,model):
    """测试集上准曲率评估"""
    data_len = len(x_)
    batch_eval = data_get.batch_iter(x_, y_, 128)
    total_loss = 0
    total_acc = 0
    for batch_xs, batch_ys in batch_eval:
        batch_len = len(batch_xs)
        # loss, acc = sess.run([model.loss, model.accuracy], feed_dict={model.x: batch_xs, model.y_: batch_ys})
        model.lr(batch_xs, batch_ys)
        loss, acc = model.loss, model.accuracy
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss/data_len, total_acc/data_len


def get_data():
    # 读取数据集
    print("Loading training and validation data...")
    # data_get 在本文件 main 那里获得
    X_train, X_test, y_train, y_test = data_get.provide_data() # 卡在这一步
    print("***3***")
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    return X_train, X_test, y_train, y_test, len(X_train[0])


def train(X_train, X_test, y_train, y_test):
    # 训练模型
    print("Training and evaluating...")
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    flag = False
    # 初始化 w, b
    model = LrModel(config, seq_length)  # 修改，放到这里来，还会每次都传入w,b嘛？
    for step in range(config.num_epochs):  #step没有使用上
        print(step)
        batch_train = data_get.batch_iter(X_train, y_train)
        for batch_xs, batch_ys in batch_train:
            # TODO 这里是有很大问题的，每次调用 LrModel 都会重新初始化 w, b，之前训练存下来的优化，都没了
            # 有时间来修改这里吧，学习怎么保存模型的参数，不每次都初始化，应该能提升 ACC
            # model = LrModel(config, seq_length, batch_xs, batch_ys,w,b)
            if total_batch % config.print_per_batch == 0:
                # loss_train, acc_train = sess.run([model.loss, model.accuracy], feed_dict={model.x: X_train, model.y_: y_train}
                loss_train, acc_train = model.loss, model.accuracy
                loss_val, acc_val = evaluate(X_test, y_test, model)

                print("best_acc_val:",best_acc_val)
                if acc_val > best_acc_val:
                    print("acc_val:",acc_val)
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    # 保存模型
                    # model.save('best_model',save_format='tf', save_path=config.lr_save_path)
                    try:
                        model.save_weights("task1.h5")
                    except:
                        pass
                    """
                    保存模型为 pb格式
                    model.save('path_to_my_model',save_format='tf')
                    new_model = tf.keras.models.load_model('path_to_my_model')
                    new_model.predict(x_test)
                    """
                    improve_str = "*"
                else:
                    improve_str = ""
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, '\
                        + 'Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improve_str))
            # sess.run(model.train_step, feed_dict={model.x: batch_xs, model.y_: batch_ys})
            # 这里才是更新梯度 
            model.lr(batch_xs, batch_ys)
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                #  验证集准确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


# TODO:后续有需要再做
def test():
    """
    目前直接输入一个语料，分为训练集和验证集合
    也可以输入两个，一个训练集用sklearn分为训练集和验证集，单独找一个验证集再这测试
    还可以输入训练集、验证集、测试集，测试集在这做测试
    """
    pass


if __name__ == "__main__":
    config = LrConfig()
    data_get = DataProcess(config.dataset_path, config.stopwords_path, config.tfidf_model_save_path)
    print("***2***")
    X_train, X_test, y_train, y_test, seq_length = get_data()
    print("***1***")
    train(X_train, X_test, y_train, y_test)