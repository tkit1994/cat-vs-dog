import numpy as np
import tensorflow as tf
from itertools import count
from nets.resnet_v2 import resnet_v2_50, resnet_arg_scope


def preprocess(filename, label):
    '''
    处理图片，被dataset.map调用，此处的map和python中的map用法差不多，
    一些数据增强的操作也可以在这里面写，也可以再写个函数继续map
    '''
    # 文件的全路径
    fullpath = tf.string_join(["data/train/", filename])
    # 读文件
    img = tf.read_file(fullpath)
    # decode文件，二进制-->Tensor，用到的函数取决于文件的格式
    img = tf.image.decode_jpeg(img, channels=3)
    # resize使图片大小想用
    img = tf.image.resize_images(img, (224, 224))
    # 归一化操作，此处归一化到(-1, 1),不用归一化应该也可以，一般减去各个通道的中值，
    # 我这里偷懒没有算中值，然后归一化到(-1, 1)或者(0, 1),不归一化应该也可以，
    # caffe中好像就直接减去中值，没有进一步做处理，pytorch中是减去中值，然后归一化。
    img -= 127
    img /= 128

    return img, label


# 使用numpy取文件的label, 先以string的形式读入
labelmap = np.genfromtxt("label.csv", dtype="U", delimiter=",")

# 索引filename和label并转换为对应的数据类型
filenames = labelmap[:, 0].astype(np.unicode)
labels = labelmap[:, 1].astype(np.int64)

# 建立dataset
# shuffle为打乱的意思，打乱顺序，但是文件名和标签还是相对应的。只是读取的顺序变了
dataset = tf.data.Dataset.from_tensor_slices(
    (filenames, labels)).shuffle(len(filenames))

# num_parallel_calls: preprocess的线程数量, 此处为8个线程，可以调整
# batch(32): batchsize为32，可以调整
# prefetch(1): 预先读取1个batch, 可以加快训练，显卡一直有数据可以训练，不用等待cpu读取数据
dataset = dataset.map(preprocess, num_parallel_calls=8).batch(32).prefetch(1)

it = dataset.make_initializable_iterator()

image_batch, label_batch = it.get_next()

# arg_scope可以设置一些操作中的默认值
with tf.contrib.slim.arg_scope(resnet_arg_scope()):
    logits, endpoints = resnet_v2_50(image_batch, num_classes=2)

# 计算loss
loss = tf.losses.sparse_softmax_cross_entropy(
    labels=label_batch, logits=logits)
# 计算accuracy
correct = tf.equal(tf.argmax(logits, 1), label_batch)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# 将loss和accuracy加入summary, 通过tensorboard显示
tf.summary.scalar("train/loss", loss)
tf.summary.scalar("train/accuracy", accuracy)
merged = tf.summary.merge_all()
# global_step, 每次sess.run()会加1
global_step = tf.Variable(0, trainable=False)
# 优化器，这里使用的是adam, 可以尝试使用其它的优化器，adam比较常用
optimzer = tf.train.AdamOptimizer()

# 使用batchnorm的话要这样。
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimzer.minimize(loss, global_step=global_step)
# 定义saver用来保存模型
saver = tf.train.Saver()

# 开始训练
with tf.Session() as sess:
    # 初始化变量， 前面定义的包括网络内的变量在这里才真正开始初始化
    tf.global_variables_initializer().run()
    # summary writer, 用来在写入graph, 以及summary
    writer = tf.summary.FileWriter(logdir='./logs', graph=sess.graph)
    # 训练过程
    # 训练20个epoch
    for epoch in range(20):
        print("Epoch {}/{}".format(epoch, 20))
        # 每个epoch重新初始化Iterator
        sess.run(it.initializer)
        # 用于保存整个数据集上的accuracy
        acc_avg = 0
        # 迭代，使用itertools的cout建立一个死循环
        for step in count():
            # 使用try catch 来捕获tf.errors.OutOfRangeError用来判断数据是否完全迭代完一遍，迭代完会运行except中的内容，然后退出本层循环
            try:
                # 执行对应的操作
                myloss, acc, summary, s, _ = sess.run(
                    [loss, accuracy, merged, global_step, train_op])
                # 将当前batch的accuracy加入acc_avg, 运行完当前epoch后acc_avg会除以step, 从而得到整个数据集上的平均accuracy
                acc_avg += acc
                # 每10步显示以及保存summary
                if step % 10 == 0:
                    writer.add_summary(summary, global_step=s)
                    print("Step: {}, loss: {}, accuracy: {}".format(
                        step, myloss, acc))
            # 数据迭代完后执行这个
            except tf.errors.OutOfRangeError:
                # 打印当前epoch, accuracy 以及保存网络参数
                print("Epoch {} done!".format(epoch))
                print("accuracy: {}".format(acc_avg / step))
                saver.save(sess, "logs/resnet_v2_50.ckpt")
                # 跳出本层循环
                break
