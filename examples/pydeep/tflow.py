"""
The contents of this file have been plagiarized, with abandon,
from [tensorflow for poets](FIXME-link).

The strategy supported by this module is the following:

 1. create all the specialized graphs needed for:
    1. bottleneck calculation
    2. new traning network
 2. save the new graphs to hdfs
 3. use specialized classes to load and execute the graphs.

model structure

"""

# First pass implementation, we are not using any of the modern stuff like
# the tf.data API


import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np


def load_graph(path, return_elements):
    with tf.Graph().as_default() as graph:
        with gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            elements = tf.import_graph_def(graph_def,
                                           name='',  # disable 'import' prfx
                                           return_elements=return_elements)
    return graph, elements


def save_graph(graph, path):
    with tf.Session(graph=graph):
        output_graph_def = graph.as_graph_def(add_shapes=True)
        with gfile.FastGFile(path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


class BottleneckProjector(object):

    @classmethod
    def create_graph(cls, model):
        """Creates a new (disjoint) graph that contains: a jpeg to
        normalized data conversion; and a graph constructed by cutting
        at the bottleneck an existing, pre-trained network.
        """
        def add_jpeg_decoding():
            m = model
            jpeg_data = tf.placeholder(tf.string, name=m['jpg_input'])
            dimage = tf.image.decode_jpeg(jpeg_data, channels=m['input_depth'])
            dimage_as_float = tf.cast(dimage, dtype=tf.float32)
            dimage_4d = tf.expand_dims(dimage_as_float, 0)
            resize_shape = tf.stack([m['input_height'], m['input_width']])
            resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
            resized_image = tf.image.resize_bilinear(dimage_4d,
                                                     resize_shape_as_int)
            offset_image = tf.subtract(resized_image, m['input_mean'])
            tf.multiply(offset_image, 1.0 / m['input_std'],
                        name=m['mul_image'])

        path = model['path']
        bneck_name = model['bottleneck_tensor_name']
        input_name = model['resized_input_tensor_name']
        graph, elements = load_graph(path, [bneck_name, input_name])
        with tf.Session(graph=graph):
            add_jpeg_decoding()
        return graph

    def __init__(self, model):
        self.model = model
        graph, (input_jpeg, mul_image, input_tensor, bneck_tensor) = \
            load_graph(model['path'], [model[x]
                                       for x in ['jpg_input_tensor_name',
                                                 'mul_image_tensor_name',
                                                 'resized_input_tensor_name',
                                                 'bottleneck_tensor_name']])
        self.graph = graph
        self.input_jpeg = input_jpeg
        self.mul_image = mul_image
        self.resized_input_tensor = input_tensor
        self.bneck_tensor = bneck_tensor

    def project(self, image_path):
        with tf.Session(graph=self.graph) as s:
            jpeg_data = gfile.FastGFile(image_path, 'rb').read()
            m_idat = s.run(self.mul_image,
                           {self.input_jpeg: jpeg_data})
            b_val = s.run(self.bneck_tensor,
                          {self.resized_input_tensor: m_idat})
        return np.squeeze(b_val)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor
       (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class Retrainer(object):
    @classmethod
    def create_graph(cls, model, class_count):
        """Adds a new softmax and fully-connected layer for training"""
        with tf.Graph().as_default() as graph:
            (ground_truth_input,
             final_tensor) = cls.add_training_ops(model, class_count)
            cls.add_evaluation_step(final_tensor, ground_truth_input)
        return graph

    @classmethod
    def add_training_ops(cls, model, class_count):
        """foo """
        bneck_size = model['bottel_neck_tensor_size']
        bneck_input = tf.placeholder(
            tf.float32, shape=[None, bneck_size], name='bneck_input')
        ground_truth_input = tf.placeholder(
            tf.float32, [None, class_count], name='ground_truth_input')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # Organizing the following ops as `final_training_ops`
        # so they're easier to see in TensorBoard
        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                initial_value = tf.truncated_normal(
                    [bneck_size, class_count], stddev=0.001)
                layer_weights = tf.Variable(initial_value,
                                            name='final_weights')
                variable_summaries(layer_weights)
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]),
                                           name='final_biases')
                variable_summaries(layer_biases)
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations', logits)
        final_tensor = tf.nn.softmax(logits, name="final_tensor")
        tf.summary.histogram('activations', final_tensor)
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=ground_truth_input, logits=logits)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer.minimize(cross_entropy_mean, name="train_step")
        return ground_truth_input, final_tensor

    @classmethod
    def add_evaluation_step(cls, result_tensor, ground_truth_tensor):
        """Inserts the operations we need to evaluate the accuracy of our results.
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction = tf.argmax(result_tensor, 1)
                correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)

    def __init__(self, path):
        self.graph = load_graph(path)
        # we exploit the fact that we know what the graph should look like,
        # since it was built with our class method. Therefore, we can directly
        # extract the tensors we need.

    def run(self, xxx):
        # here we actually run the calculation
        pass
