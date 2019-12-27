import skimage
import tensorflow as tf
from skimage import io # [MUST] for skimage.io.imread
import os
import matplotlib.pyplot as plt # draw distribution graph
from skimage import transform
from skimage.color import rgb2gray # convert img to grayscale
import numpy as np
from tensorflow.contrib import layers

def first_try():
    # initialize constant
    x1 = tf.constant([1,2,3,4])
    x2 = tf.constant([5,6,7,8])
    # multiply
    result = tf.multiply(x1, x2)
    # only return a tensor, not real-value
    # that means: tf does not calculate. only deprive a graph
    print(result) # Tensor("Mul:0", shape=(4,), dtype=int32)
    # run result and print. 'with' will close automatically
    #sess = tf.Session()
    #print(sess.run(result))
    #sess.close()
    with tf.Session() as sess:
        output = sess.run(result)
        print(output)

def load_data(data_dir):
    dirs = [d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    images = []
    # each type of sign
    for d in dirs:
        # .ppm 's file name
        label_dir = os.path.join(data_dir, d)
        # real path of .ppm
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm")]
        for f in file_names:
            # load image
            images.append(skimage.io.imread(f))
            labels.append(int(d))
    return images, labels

def random_show(images, name, cmap=None):
    for i in range(len(name)):
        plt.subplot(1, len(name), i+1)
        plt.axis('off')
        # add cmap for gray-scaled pic, which set cmap='gray'
        # or u'll get wrong color
        plt.imshow(images[name[i]], cmap)
        plt.subplots_adjust(wspace=0.5)
        print("shape: {0}, min: {1}, max: {2}".format(images[name[i]].shape,
                                                      images[name[i]].min(),
                                                      images[name[i]].max()))
    plt.show()


def show_each_label_pic(labels):
    uniq_labels = set(labels)
    # initialize the figure
    plt.figure(figsize=(15, 15))
    i = 1
    for label in uniq_labels:
        # pick the 1st image for each label
        image = images[labels.index(label)]
        # 8X8, ith
        plt.subplot(8, 8, i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image) # plot single picture
    plt.show()

def transform_img(images, rows, cols):
    return [transform.resize(image, (rows, cols)) for image in images]

def to_gray(images):
    # need array
    return rgb2gray(np.array(images))

#def create_model(images_flat, label_num):



if __name__=="__main__":
    ROOT_PATH = r"G:/share/testTF"
    train_data_dir = ROOT_PATH + "/Training"
    images, labels = load_data(train_data_dir)
    #print(len(set(labels))) # 62. coz 62 type of traffic signs
    #print(len(images)) # 4575
    #plt.hist(labels, 63) # draw a bar-graph.
    #plt.show()
    #random_show(images, [300, 2250, 3650, 4000])
    #print(type(images[0])) # <class 'numpy.ndarray'>
    #show_each_label_pic(labels)
    images28 = transform_img(images, 28, 28)
    #random_show(images28, [300, 2250, 3650, 4000])
    gray_images28 = to_gray(images28)
    #random_show(gray_images28, [300, 2250, 3650, 4000], cmap="gray")
    # create models
    # placeholder will be initialized by session. Need run(). Empty now.
    # x are imgs
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
    # y are labels
    y = tf.placeholder(dtype=tf.int32, shape=[None])
    # flatten, -> [None, 28*28]
    images_flat = layers.flatten(x)
    # fully-connected
    logits = layers.fully_connected(images_flat, 62, tf.nn.relu)
    # define a loss function. 每个对象只有一个label
    # only takes labels like int32, int64
    # tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y2) 接受one-hot数字标签
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    # define an optimizer. eg: stochastic gradient descent(SGD), ADAM, RMSprop
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # convert logits to label indexes
    correct_pred = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # create model ends. run now
    tf.set_random_seed(1234)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: gray_images28, y:labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print("DONE WITH EPOCH")

    #sess.close()
    # test now
    test_data_dir = ROOT_PATH + "/Testing"
    test_images, test_labels = load_data(test_data_dir)
    test_images28 = transform_img(test_images, 28, 28)
    test_gray_images28 = to_gray(test_images28)
    predicted = sess.run([correct_pred], feed_dict={x: test_gray_images28})[0]
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    test_accuracy = match_count*1.0 / len(test_labels)
    print("Test Accuracy: {:.3f}".format(test_accuracy)) # use gray_images28 : 0.662 0.489 0.573 0.696 ...
    sess.close()


