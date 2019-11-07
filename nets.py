import copy

import tensorflow.compat.v1 as tf
import numpy as np

from helmo.util import tensor_ops


class Network:
    def __init__(
            self,
            config,
    ):
        self._config = copy.deepcopy(config)
        self._num_nodes = self._config['num_nodes']
        self._opt_type = self._config['optimizer']
        self._init_parameter = config['init_parameter']
        self.init_ops = None
        self.fetches = {'metrics': {}, 'tensors': {}}
        self.datasets = None
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

    def build(self, datasets):
        self.datasets = datasets
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
        loss = tf.reduce_mean(cr_entr)
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
            'tensors': self.fetches['tensors'],
        }


class DenseNetwork(Network):
    def __init__(
            self,
            config,
    ):
        super().__init__(config)

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
        activations = [tf.nn.relu] * (len(self._config['num_nodes'])-1) + [None]
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
            self.fetches['tensors'][tensor_name] = tensor_ops.corcov_loss(
                hs,
                reduced_axes=[0],
                cor_axis=1,
                punish='correlation',
                reduction='mean',
                norm='sqr'
            )
            tensor_name = 'hs{}_rms'.format(i)
            self.fetches['tensors'][tensor_name] = tf.sqrt(
                tf.reduce_mean(tf.square(hs)))
            self.layers.append(layer)
        return hs


class Conv2dNetwork(Network):
    def __init__(self, config):
        super().__init__(config)

    def _prep_inputs_labels(self):
        inputs, labels = self.iterator.get_next()
        # print(self.datasets['valid'].element_spec)
        inputs = tf.cast(inputs, tf.float32)
        labels = tf.one_hot(labels, 10, dtype=tf.float32)
        if self._config['shuffle']:
            inputs = shuffle(inputs)
        return inputs, labels

    def _net(self, inputs):
        self.layers = []
        hs = inputs
        input_shape = tf.data.get_output_shapes(
            list(self.datasets.values())[0]).as_list()
        for i, layer_specs in enumerate(self._config['layers']):
            layer_specs = copy.deepcopy(layer_specs)
            layer_type = layer_specs["type"]
            LayerClass = get_layer_class(layer_specs)
            layer_specs = prepare_layer_specs(
                layer_specs,
                self._init_parameter
                / (input_shape[1] * input_shape[2] * input_shape[3]) ** 0.5
            )
            layer = LayerClass(**layer_specs)
            hs = layer(hs)
            if layer_type != 'flatten':
                if layer_type == 'conv_2d' or 'max_pooling_2d':
                    reduced_axes = [0, 1, 2]
                elif layer_type == 'dense':
                    reduced_axes = [0]
                tensor_name = 'hs{}_corr'.format(i)
                self.fetches['tensors'][tensor_name] = tensor_ops.corcov_loss(
                    hs,
                    reduced_axes=reduced_axes,
                    cor_axis=3,
                    punish='correlation',
                    reduction='mean',
                    norm='sqr'
                )
                tensor_name = 'hs{}_rms'.format(i)
                self.fetches['tensors'][tensor_name] = tf.sqrt(
                    tf.reduce_mean(tf.square(hs)))
                self.layers.append(layer)
                input_shape = hs.get_shape().as_list()
            # input_shape = get_out_shape(input_shape, layer_type, layer_specs)
        return hs


def get_input_size(dataset):
    shapes = tf.data.get_output_shapes(dataset)
    shape = shapes[0].as_list()
    return np.prod(shape[1:])


def shuffle(tensor):
    tensor = tf.transpose(tensor)
    tensor = tf.random.shuffle(tensor)
    return tf.transpose(tensor)


def get_out_shape(input_shape, layer_type, layer_specs):
    supported_layer_types = ['conv_2d', 'flatten', 'dense', 'max_pooling_2d']
    if layer_type == 'conv_2d':
        input_shape = get_conv_2d_out_shape(
            input_shape,
            layer_specs['filters'],
            layer_specs['kernel_size'],
            layer_specs['strides'],
            layer_specs['padding']
        )
    elif layer_type == 'max_pooling_2d':
        input_shape = get_max_pooling_2d_out_shape(
            input_shape,
            layer_specs['pool_size'],
            layer_specs['strides'],
            layer_specs['padding']
        )
    elif layer_type == 'flatten':
        input_shape = [
            input_shape[0],
            input_shape[1] * input_shape[2] * input_shape[3]
        ]
    elif layer_type == 'dense':
        input_shape = [input_shape[0], layer_specs['units']]
    else:
        raise ValueError(
            "Provided layer type {} is not in list of supported "
            "layer types {}".format(repr(layer_type), supported_layer_types)
        )
    return input_shape


def get_max_pooling_2d_out_shape(input_shape, pool_size, strides, padding):
    if len(input_shape) != 4:
        raise ValueError(
            "The input of 2D max pooling layer has to have 4 dimensions "
            "whereas\n"
            "input_shape == {}".format(input_shape))
    if len(pool_size) != 2:
        raise ValueError(
            "The pool size of 2D max pooling layer has to have 2 "
            "dimensions whereas\n"
            "pool_size == {}".format(pool_size)
        )
    if len(strides) != 2:
        raise ValueError(
            "The strides of 2D max pooling layer has to have 2 "
            "dimensions whereas\n"
            "strides == {}".format(strides)
        )
    res = [input_shape[0]]
    for inp, k, s in zip(input_shape[1:], pool_size, strides):
        if padding == 'valid':
            inp -= k - 1
        res.append(inp // s)
    res.append(input_shape[-1])
    return res


def get_conv_2d_out_shape(inp_shape, filters, kernel_size, strides, padding):
    if len(inp_shape) != 4:
        raise ValueError(
            "The input of 2D convolutional layer has to have 4 dimensions "
            "whereas\n"
            "input_shape == {}".format(inp_shape))
    if len(kernel_size) != 2:
        raise ValueError(
            "The kernel size of 2D convolutional layer has to have 2 "
            "dimensions whereas\n"
            "kernel_size == {}".format(kernel_size)
        )
    if len(strides) != 2:
        raise ValueError(
            "The strides of 2D convolutional layer has to have 2 "
            "dimensions whereas\n"
            "strides == {}".format(strides)
        )
    res = [inp_shape[0]]
    for inp, k, s in zip(inp_shape[1:], kernel_size, strides):
        if padding == 'valid':
            inp -= k - 1
        res.append(inp // s)
    res.append(filters)
    return res


def get_layer_class(specs):
    supported_layer_types = ['conv_2d', 'flatten', 'dense', 'max_pooling_2d']
    if specs['type'] == 'conv_2d':
        Class = tf.layers.Conv2d
    elif specs['type'] == 'flatten':
        Class = tf.layers.Flatten
    elif specs['type'] == 'dense':
        Class = tf.layers.Dense
    elif specs['type'] == 'max_pooling_2d':
        Class = tf.layers.MaxPooling2D
    else:
        raise ValueError(
            "Provided layer type {} is not in list of \nsupported "
            "layer types {}".format(repr(specs['type']), supported_layer_types)
        )
    return Class


def prepare_layer_specs(specs, init_stddev):
    supported_activations = ['relu']
    supported_kernel_initializers = ['truncated_normal']
    specs = copy.deepcopy(specs)
    del specs['type']
    if 'activation' in specs:
        if specs['activation'] == 'relu':
            specs['activation'] = tf.nn.relu
        else:
            raise ValueError(
                "Provided activation {} is not in list of "
                "\nsupported activations {}".format(
                    repr(specs['activation']), supported_activations)
            )
    if 'kernel_initializer' in specs:
        if specs['kernel_initializer'] == 'truncated_normal':
            specs['kernel_initializer'] = tf.truncated_normal_initializer(
                0,
                init_stddev
            )
        else:
            raise ValueError(
                "Provided kernel initializer {} is not in list of\n"
                "supported kernel initializers {}".format(
                    repr(specs['kernel_initializer']),
                    supported_kernel_initializers)
            )
    return specs
