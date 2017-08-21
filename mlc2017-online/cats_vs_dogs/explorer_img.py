
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


file = 'C:\\Users\\pjw92\\PycharmProjects\\mlc2017-online\\cats_vs_dogs\\train\\train-00001-of-00014'

fig = plt.figure()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
record_iterator = tf.python_io.tf_record_iterator(path=file)
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    img = example.features.feature["image/encoded"].bytes_list.value[0]
    tfimage = tf.image.decode_jpeg(img, channels=3)
    im = sess.run(tfimage)
    plt.imshow(im)
    plt.show()
    tfimage = tf.expand_dims(tfimage, 0)
    tfimage = tf.image.resize_images(tfimage, [32, 32])
