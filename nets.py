import copy
import time

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import numpy as np

from helmo.util import tensor_ops


def calculate_ds_moments(ds, ds_size):
    s = 0
    s_sqr = 0
    n = 0
    num_elements = 0
    ds = tfds.as_numpy(ds)
    while num_elements < ds_size:
        inp = next(ds)[0]
        s += np.sum(inp)
        s_sqr += np.sum(inp**2)
        n += inp.size
        num_elements += inp.shape[0]
    mean = s / n
    sqr_mean = s_sqr / n
    var = (sqr_mean - mean**2) * n / (n-1)
    return mean, var


def get_normal_real_eigen_initializer(n):
    d = np.random.randn(n)
    eigen_vectors_basis_m = np.diag(d)
    eigen_vectors = np.random.randn(n, n)
    norms = np.linalg.norm(eigen_vectors, axis=0, keepdims=True)
    eigen_vectors /= norms
    inv = np.linalg.inv(eigen_vectors)
    matrix = inv @ eigen_vectors_basis_m @ eigen_vectors / 60
    return tf.initializers.constant(matrix)


def get_diag_initializer(n):
    d = np.random.randn(n)
    return tf.initializers.constant(np.diag(d))


class Network:
    def __init__(
            self,
            config,
    ):
        self._config = copy.deepcopy(config)
        self._opt_type = self._config['optimizer']
        self._init_parameter = config['init_parameter']
        self.l2_reg_coef = config['l2_reg_coef']
        self.init_ops = None
        self.fetches = {'accumulators': {}, 'tensors': {}, 'metrics': {}}
        self.datasets = None
        self.datasets_sizes = None
        self.iterator = None
        self.layers = None
        self.input_shape = None
        self.opt = None
        self.feed_dict = {}
        self.train_op = None

    def _create_iterator(self, datasets):
        ds = list(datasets.values())[0]
        self.iterator = tf.data.Iterator.from_structure(
            ds.output_types, ds.output_shapes)

    def _create_init_ops(self, datasets):
        init_ops = {}
        for ds_name, ds in datasets.items():
            init_ops[ds_name] = self.iterator.make_initializer(ds)
        self.init_ops = init_ops

    def _create_placeholders(self):
        self.feed_dict['lr'] = tf.placeholder(tf.float32)

    def _create_optimizer(self):
        if self._opt_type == 'sgd':
            opt_class = tf.train.GradientDescentOptimizer
        elif self._opt_type == 'adam':
            opt_class = tf.train.AdamOptimizer
        else:
            raise ValueError(
                "Only 'sgd' and 'adam' optimizer types are supported")
        self.opt = opt_class(self.feed_dict['lr'])

    def build(self, datasets, datasets_sizes):
        self.datasets = datasets
        self.datasets_sizes = datasets_sizes
        self._create_iterator(datasets)
        self._create_init_ops(datasets)
        self._create_placeholders()
        inputs, labels = self._prep_inputs_labels()
        logits = self._net(inputs)
        preds = tf.nn.softmax(logits)
        cr_entr = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=labels
        )
        loss = tf.reduce_mean(cr_entr) + self.get_l2_loss()
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(labels, axis=-1),
                    tf.argmax(preds, axis=-1)
                ),
                dtype=tf.float32
            )
        )
        self.fetches['metrics']['loss'] = loss
        self.fetches['metrics']['acc'] = acc
        self.fetches['preds'] = preds
        self._create_optimizer()
        self.train_op = self.opt.minimize(loss)
        self.fetches['train'] = {
            "train_op": self.train_op,
            "metrics": self.fetches['metrics'],
        }
        self.fetches['valid'] = {
            "metrics": self.fetches['metrics'],
            'accumulators': self.fetches['accumulators'],
        }

    def get_l2_loss(self):
        loss = 0
        if self.l2_reg_coef is None or self.l2_reg_coef == 0:
            return 0
        for layer in self.layers:
            if hasattr(layer, 'kernel'):
                loss += tf.nn.l2_loss(layer.kernel)
        return self.l2_reg_coef * loss


class DenseNetwork(Network):
    def __init__(
            self,
            config,
    ):
        super().__init__(config)
        self._num_nodes = self._config['num_nodes']

    def _prep_inputs_labels(self):
        inputs, labels = self.iterator.get_next()
        # print(self.datasets['valid'].element_spec)
        inputs = tf.cast(inputs, tf.float32)
        labels = tf.one_hot(labels, 10, dtype=tf.float32)
        sh = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1, np.prod(sh[1:])])
        if self._config['shuffle']:
            inputs = shuffle(inputs)
        return inputs, labels

    def _net(self, inputs):
        self.layers = []
        activations = [tf.nn.relu] \
            * (len(self._config['num_nodes'])-1) + [None]
        hs = inputs
        input_sizes = [get_input_size(list(self.datasets.values())[0])] \
            + self._config['num_nodes'][:-1]
        for i, nn in enumerate(self._config['num_nodes']):
            layer = tf.layers.Dense(
                nn,
                activation=activations[i],
                kernel_initializer=tf.truncated_normal_initializer(
                    0,
                    self._init_parameter / (nn + input_sizes[i])**0.5,
                )
            )
            hs = layer(hs)
            tensor_name = 'hs{}_corr'.format(i)
            self.fetches['accumulators'][tensor_name] = tensor_ops.corcov_loss(
                hs,
                reduced_axes=[0],
                cor_axis=1,
                punish='correlation',
                reduction='mean',
                norm='sqr'
            )
            tensor_name = 'hs{}_rms'.format(i)
            self.fetches['accumulators'][tensor_name] = tf.sqrt(
                tf.reduce_mean(tf.square(hs)))
            tensor_name = 'kernel{}'.format(i)
            self.fetches['tensors'][tensor_name] = layer.kernel
            self.layers.append(layer)
        return hs


class Conv2dNetwork(Network):
    def __init__(self, config):
        super().__init__(config)

    def _prep_inputs_labels(self):
        inputs, labels = self.iterator.get_next()
        # print(self.datasets['valid'].element_spec)
        inputs = tf.cast(inputs, tf.float32)
        mean, variance = calculate_ds_moments(
            self.datasets['train'], self.datasets_sizes['train'])
        inputs = (inputs - mean) / variance**0.5
        labels = tf.one_hot(labels, 10, dtype=tf.float32)
        if self._config['shuffle']:
            inputs = shuffle(inputs)
        return inputs, labels

    def _net(self, inputs):
        self.layers = []
        hs = inputs
        input_shape = tf.data.get_output_shapes(
            list(self.datasets.values())[0])[0].as_list()
        for i, layer_specs in enumerate(self._config['layers']):
            layer_specs = copy.deepcopy(layer_specs)
            layer_type = layer_specs["type"]
            LayerClass = get_layer_class(layer_specs)
            layer_specs = prepare_layer_specs(
                layer_specs,
                input_shape,
                self._init_parameter
            )
            layer = LayerClass(**layer_specs)
            self.layers.append(layer)
            hs = layer(hs)
            if layer_type not in ['flatten', 'batch_norm']:
                if layer_type in ['conv_2d', 'max_pooling_2d']:
                    reduced_axes = [0, 1, 2]
                    cor_axis = 3
                elif layer_type == 'dense':
                    reduced_axes = [0]
                    cor_axis = 1
                else:
                    raise ValueError(
                        "Unsupported layer type {}\n"
                        "Supported layer types for "
                        "correlation computation: {}".format(
                            layer_type,
                            ['dense', 'conv_2d', 'max_pooling_2d']
                        )
                    )
                tensor_name = 'hs{}_corr'.format(i)
                self.fetches['accumulators'][tensor_name] = \
                    tensor_ops.corcov_loss(
                        hs,
                        reduced_axes=reduced_axes,
                        cor_axis=cor_axis,
                        punish='correlation',
                        reduction='mean',
                        norm='sqr'
                    )
                tensor_name = 'hs{}_rms'.format(i)
                self.fetches['accumulators'][tensor_name] = tf.sqrt(
                    tf.reduce_mean(tf.square(hs)))
                tensor_name = 'kernel{}'.format(i)
                self.fetches['tensors'][tensor_name] = layer.kernel
            input_shape = hs.get_shape().as_list()
        return hs


def get_input_size(dataset):
    shapes = tf.data.get_output_shapes(dataset)
    shape = shapes[0].as_list()
    return np.prod(shape[1:])


def shuffle(tensor):
    tensor = tf.transpose(tensor)
    tensor = tf.random.shuffle(tensor)
    return tf.transpose(tensor)


def get_layer_class(specs):
    supported_layer_types = [
        'conv_2d', 'flatten', 'dense', 'max_pooling_2d', 'batch_norm']
    if specs['type'] == 'conv_2d':
        Class = tf.layers.Conv2D
    elif specs['type'] == 'flatten':
        Class = tf.layers.Flatten
    elif specs['type'] == 'dense':
        Class = tf.layers.Dense
    elif specs['type'] == 'max_pooling_2d':
        Class = tf.layers.MaxPooling2D
    elif specs['type'] == 'batch_norm':
        Class = tf.layers.BatchNormalization
    else:
        raise ValueError(
            "Provided layer type {} is not in list of \nsupported "
            "layer types {}".format(repr(specs['type']), supported_layer_types)
        )
    return Class


def prepare_layer_specs(specs, input_shape, init_parameter):
    supported_activations = ['relu', 'leaky_relu']
    supported_kernel_initializers = ['truncated_normal', 'real_eigen', 'diag']
    specs = copy.deepcopy(specs)
    del specs['type']
    if 'activation' in specs:
        if specs['activation'] == 'relu':
            specs['activation'] = tf.nn.relu
        elif specs['activation'] == 'leaky_relu':
            specs['activation'] = tf.nn.leaky_relu
        else:
            raise ValueError(
                "Provided activation {} is not in list of "
                "\nsupported activations {}".format(
                    repr(specs['activation']), supported_activations)
            )
    if 'kernel_initializer' in specs:
        if len(input_shape) == 4:
            inp_dim = input_shape[1] * input_shape[2] * input_shape[3]
            output_dim = specs['filters']
        elif len(input_shape) == 2:
            inp_dim = input_shape[1]
            output_dim = specs['units']
        else:
            raise ValueError(
                "Unsupported input shape {}. "
                "Input shape has to have 2 or 4 dimensions.".format(
                    input_shape)
            )

        if specs['kernel_initializer'] == 'truncated_normal':
            specs['kernel_initializer'] = tf.truncated_normal_initializer(
                0,
                init_parameter / (inp_dim + output_dim) ** 0.5
            )
        elif specs['kernel_initializer'] == 'real_eigen':
            specs['kernel_initializer'] = get_normal_real_eigen_initializer(
                specs['units']
            )
        elif specs['kernel_initializer'] == 'diag':
            specs['kernel_initializer'] = get_diag_initializer(
                specs['units'])
        else:
            raise ValueError(
                "Provided kernel initializer {} is not in list of\n"
                "supported kernel initializers {}".format(
                    repr(specs['kernel_initializer']),
                    supported_kernel_initializers)
            )
    return specs
