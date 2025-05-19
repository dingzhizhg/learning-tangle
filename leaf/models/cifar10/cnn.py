import tensorflow as tf

from model import Model
# from ..utils.tf_utils import graph_size
# from ..baseline_constants import ACCURACY_KEY
# from ...lab.dataset import batch_data
import numpy as np


IMAGE_SIZE = 32
DROPOUT = 0.35


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, optimizer=None):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr, optimizer)

    def create_model(self):
        """Model function for CNN."""
        with self.graph.as_default():
            features = tf.placeholder(
                tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='features')
            labels = tf.placeholder(tf.int64, shape=[None], name='labels')

            # 第一个卷积块
            conv1 = tf.layers.conv2d(
                inputs=features,
                filters=64,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu)
            conv1 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            pool1 = tf.layers.dropout(pool1, rate=DROPOUT)

            # 第二个卷积块
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=128,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(
                inputs=conv2,
                filters=128,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            pool2 = tf.layers.dropout(pool2, rate=DROPOUT)

            # 第三个卷积块
            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=256,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(
                inputs=conv3,
                filters=256,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
            pool3 = tf.layers.dropout(pool3, rate=DROPOUT)

            # 第四个卷积块
            conv4 = tf.layers.conv2d(
                inputs=pool3,
                filters=512,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu)
            conv4 = tf.layers.conv2d(
                inputs=conv4,
                filters=512,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
            pool4 = tf.layers.dropout(pool4, rate=DROPOUT)

            # 全连接层
            flatten = tf.layers.flatten(pool4)
            dense1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
            dense1 = tf.layers.dropout(dense1, rate=DROPOUT)
            
            dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)
            dense2 = tf.layers.dropout(dense2, rate=DROPOUT)
            
            dense3 = tf.layers.dense(inputs=dense2, units=256, activation=tf.nn.relu)
            dense3 = tf.layers.dropout(dense3, rate=DROPOUT)

            # 输出层
            logits = tf.layers.dense(inputs=dense3, units=self.num_classes)
            
            predictions = {
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            
            # 添加L2正则化
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss = loss + 0.0001 * l2_loss
            
            # 确保在同一个图中创建优化器
            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=self.global_step)
            
            eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
            conf_matrix = tf.math.confusion_matrix(labels, predictions["classes"], num_classes=self.num_classes)
            
            return features, labels, train_op, eval_metric_ops, conf_matrix, loss

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
