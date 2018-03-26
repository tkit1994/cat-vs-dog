import tensorflow as tf

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


def preprocess_test(filename):
    '''
    处理图片，被dataset.map调用，此处的map和python中的map用法差不多，
    一些数据增强的操作也可以在这里面写，也可以再写个函数继续map
    '''
    # 文件的全路径
    fullpath = tf.string_join([filename,])
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

    return img