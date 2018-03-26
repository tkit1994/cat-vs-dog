from itertools import count

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from nets.resnet_v2 import resnet_arg_scope, resnet_v2_50
from utils import preprocess
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser("A script to train resnet_2_50")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--numepochs", type=int, default=20)
    parser.add_argument("--testsize", type=float, default=0.2)
    parser.add_argument("--labelmap", type=str, default="label.csv")
    parser.add_argument("--numthreads", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="logs")
    return parser.parse_args()
def main(args):
    # 使用numpy取文件的label, 先以string的形式读入
    labelmap = np.genfromtxt(args.labelmap, dtype="U", delimiter=",")

    # 索引filename和label并转换为对应的数据类型
    filenames = labelmap[:, 0].astype(np.unicode)
    labels = labelmap[:, 1].astype(np.int64)

    # 分训练集和验证集
    filenames_train, filenames_val, labels_train, labels_val = train_test_split(
        filenames, labels, test_size=args.testsize)

    # 建立dataset
    # shuffle为打乱的意思，打乱顺序，但是文件名和标签还是相对应的。只是读取的顺序变了
    # train dataset
    dataset_train = tf.data.Dataset.from_tensor_slices(
        (filenames_train, labels_train)).shuffle(len(filenames_train))

    # num_parallel_calls: preprocess的线程数量, 此处为8个线程，可以调整
    # batch(32): batchsize为32，可以调整
    # prefetch(1): 预先读取1个batch, 可以加快训练，显卡一直有数据可以训练，不用等待cpu读取数据
    dataset_train = dataset_train.map(
        preprocess, num_parallel_calls=args.numthreads).batch(args.batchsize).prefetch(1)
    # val dataset
    dataset_val = tf.data.Dataset.from_tensor_slices(
        (filenames_val, labels_val)).shuffle(len(filenames_val))
    dataset_val = dataset_val.map(
        preprocess, num_parallel_calls=args.numthreads).batch(args.batchsize).prefetch(1)

    # 建立 Iterator
    iterator = tf.data.Iterator.from_structure(
        dataset_train.output_types, dataset_train.output_shapes)
    training_init_op = iterator.make_initializer(dataset_train)
    validation_init_op = iterator.make_initializer(dataset_val)

    image_batch, label_batch = iterator.get_next()
    istrain = tf.placeholder(tf.bool, name="istrain")
    # arg_scope可以设置一些操作中的默认值
    with tf.contrib.slim.arg_scope(resnet_arg_scope()):
        logits, endpoints = resnet_v2_50(
            image_batch, num_classes=2, is_training=istrain)

    # 计算loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=label_batch, logits=logits)

    # 计算accuracy
    correct = tf.equal(tf.argmax(logits, 1), label_batch)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 将loss和accuracy加入summary, 通过tensorboard显示
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged = tf.summary.merge_all()

    # global_step, 每次sess.run()会加1
    global_step = tf.Variable(0, trainable=False, name="global_step")

    # 优化器，这里使用的是adam, 可以尝试使用其它的优化器，adam比较常用
    optimzer = tf.train.AdamOptimizer()

    # 使用batchnorm的话要这样。
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimzer.minimize(loss, global_step=global_step)

    # 定义saver用来保存模型
    saver = tf.train.Saver(max_to_keep=None)

    # 开始训练
    with tf.Session() as sess:

        # 初始化变量， 前面定义的包括网络内的变量在这里才真正开始初始化
        tf.global_variables_initializer().run()

        # summary writer, 用来在写入graph, 以及summary
        train_writer = tf.summary.FileWriter(
            logdir=os.path.join(args.logdir, "train"), graph=sess.graph)

        # 训练过程

        # 训练20个epoch
        for epoch in range(args.numepochs):
            print("Epoch {}/{}".format(epoch, args.numepochs))
            for mode in ["train", "val"]:

                # 初始化Iterator
                if mode == "train":
                    sess.run(training_init_op)
                else:
                    sess.run(validation_init_op)

                # 用于保存整个数据集上的accuracy
                acc_avg = 0

                # 迭代，使用itertools的cout建立一个死循环
                for step in count():

                    # 使用try catch 来捕获tf.errors.OutOfRangeError用来判断数据是否完全迭代完一遍，迭代完会运行except中的内容，然后退出本层循环
                    try:
                        # 执行对应的操作
                        if mode == "train":
                            myloss, acc, summary, _ = sess.run(
                                [loss, accuracy, merged, train_op], feed_dict={istrain: True})
                            train_writer.add_summary(summary, step)
                        else:
                            myloss, acc = sess.run(
                                [loss, accuracy], feed_dict={istrain: False})
                        # 将当前batch的accuracy加入acc_avg, 运行完当前epoch后acc_avg会除以step, 从而得到整个数据集上的平均accuracy
                        acc_avg += acc
                        # 每10步显示以及保存summary
                        if step % 10 == 0:
                            print("mode: {}, step: {}, loss: {}, accuracy: {}".format(mode,
                                step, myloss, acc))
                    # 数据迭代完后执行这个
                    except tf.errors.OutOfRangeError:
                        # 打印当前epoch, accuracy 以及保存网络参数
                        print("{} Epoch {} done!".format(mode, epoch))
                        print("accuracy: {}".format(acc_avg / step))
                        if mode == "train":
                            saver.save(sess, os.path.join(args.logdir, "resnet_2_50.ckpt"))
                        # 跳出本层循环
                        break
if __name__ == "__main__":
    args = parse_args()
    main(args)