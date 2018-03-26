import csv
import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from nets.resnet_v2 import resnet_arg_scope, resnet_v2_50
from utils import preprocess_test

filenameList = glob.glob("data/test1/*.jpg")
filenameList = sorted(filenameList, key=lambda x: int(
    os.path.basename(x).split(".")[0]))

dataset = tf.data.Dataset.from_tensor_slices(
    filenameList).map(preprocess_test).batch(32).prefetch(1)
it = dataset.make_one_shot_iterator()
image_batch = it.get_next()

with tf.contrib.slim.arg_scope(resnet_arg_scope()):
    logits, _ = resnet_v2_50(
        image_batch, num_classes=2, is_training=False)
prediction = tf.arg_max(logits, 1)
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint("logs")

predList = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, checkpoint)
    while True:
        try:
            predList.append(prediction.eval())
        except tf.errors.OutOfRangeError:
            break
result = np.concatenate(predList, axis=0)
with open("result.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for i, label in enumerate(result):
        writer.writerow([i + 1, label])
