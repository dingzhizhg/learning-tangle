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
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        
        # 第一个卷积块
        conv1 = tf.layers.conv2d(
            inputs=features,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=None)
        bn1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = tf.nn.relu(bn1)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        dropout1 = tf.layers.dropout(pool1, rate=0.25)
        
        # 第二个卷积块
        conv2 = tf.layers.conv2d(
            inputs=dropout1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=None)
        bn2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = tf.nn.relu(bn2)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        dropout2 = tf.layers.dropout(pool2, rate=0.25)
        
        # 第三个卷积块
        conv3 = tf.layers.conv2d(
            inputs=dropout2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=None)
        bn3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = tf.nn.relu(bn3)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        dropout3 = tf.layers.dropout(pool3, rate=0.25)
        
        # 全连接层
        pool3_flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])
        dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=None)
        bn4 = tf.layers.batch_normalization(dense1, training=True)
        dense1 = tf.nn.relu(bn4)
        dropout4 = tf.layers.dropout(dense1, rate=0.5)
        
        logits = tf.layers.dense(inputs=dropout4, units=self.num_classes)
        
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        
        # 添加L2正则化
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = loss + 0.0001 * l2_loss
        
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=self.global_step)
        
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        conf_matrix = tf.math.confusion_matrix(labels, predictions["classes"], num_classes=self.num_classes)
        
        return features, labels, train_op, eval_metric_ops, conf_matrix, loss

    def process_x(self, raw_x_batch):
        """处理输入数据，确保形状正确"""
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        """处理标签数据"""
        return np.array(raw_y_batch)
