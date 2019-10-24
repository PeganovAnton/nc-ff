import argparse
import copy
import datetime
import json
import multiprocessing as mp
import os
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from helmo.util import tensor_ops


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Path to json file with config."
    )
    return parser.parse_args()


def split_path_into_components(path):
    result = []
    while path and path != '/':
        path, fn = os.path.split(path)
        result.append(fn)
    if path == '/':
        result.append(path)
    return result[::-1]


def get_repo_root(name):
    full_path = os.path.abspath(os.path.expanduser(__file__))
    parts = split_path_into_components(full_path)
    idx = parts.index(name)
    repo_root = os.path.join(*parts[:idx+1])
    return repo_root


def get_save_path(config_path, repo_name, path_to_configs, path_to_results):
    config_path = os.path.expanduser(config_path)
    full_path = os.path.abspath(config_path)
    full_path = os.path.splitext(full_path)[0]
    path_parts = split_path_into_components(full_path)
    repo_idx = path_parts.index(repo_name)
    path_to_configs = split_path_into_components(path_to_configs)
    rel_to_results_idx = repo_idx + 1 + len(path_to_configs)
    path_in_repo = path_parts[rel_to_results_idx:]
    save_path = os.path.join(
        *path_parts[:repo_idx],
        repo_name,
        path_to_results,
        *path_in_repo
    )
    return save_path


def logarithmic_int_range(start, stop, factor, include_stop=False):
    steps = []
    while start < stop:
        steps.append(start)
        if int(start * factor) <= start:
            start += 1
        else:
            start *= factor
            start = int(start) + int(start - int(start) > 0)
    if include_stop:
        steps.append(stop)
    return steps


def get_input_size(dataset):
    shapes = tf.data.get_output_shapes(dataset)
    shape = shapes[0].as_list()
    prod = 1
    for dim in shape[1:]:
        prod *= dim
    return prod


def shuffle(tensor):
    tensor = tf.transpose(tensor)
    tensor = tf.random.shuffle(tensor)
    return tf.transpose(tensor)


class DenseNetwork:
    def __init__(
            self,
            config,
    ):
        self._config = config
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

    def _create_init_ops(self, datasets):
        init_ops = {}
        for ds_name, ds in datasets.items():
            init_ops[ds_name] = self.iterator.make_initializer(ds)
        self.init_ops = init_ops

    def _create_iterator(self, datasets):
        ds = list(datasets.values())[0]
        self.iterator = tf.data.Iterator.from_structure(
            ds.output_types, ds.output_shapes)

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
        activations = [tf.nn.relu, None]
        hs = inputs
        input_sizes = [
            get_input_size(list(self.datasets.values())[0]),
            self._num_nodes[0]
        ]
        for i, nn in enumerate(self._num_nodes):
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


def log(step, tr_metrics, vd_metrics):
    tmpl = "{time} STEP: {step} {valid}"
    msg = tmpl.format(
        time=datetime.datetime.now(), step=step, valid=vd_metrics)
    print(msg)


def accumulate(accumulator, addition):
    for k, v in addition.items():
        if k in accumulator:
            accumulator[k][0] += v
            accumulator[k][1] += 1
        else:
            accumulator[k] = [v, 1]
    return accumulator


def average(accumulator):
    mean = copy.deepcopy(accumulator)
    for k, v in mean.items():
        mean[k] = mean[k][0] / mean[k][1]
    return mean


def save_metrics(step, metrics, save_path):
    os.makedirs(save_path, exist_ok=True)
    for k, v in metrics.items():
        fn = os.path.join(save_path, k + '.txt')
        with open(fn, 'a') as f:
            if step is None:
                f.write("{}\n".format(v))
            else:
                f.write("{} {}\n".format(step, v))


def save_tensors(tensors, save_path):
    os.makedirs(save_path, exist_ok=True)
    for k, v in tensors.items():
        fn = os.path.join(save_path, k + '.pickle')
        with open(fn, 'ab') as f:
            pickle.dump(v, f)


def test(sess, train_step, model, mode, save_path):

    sess.run(model.init_ops[mode])
    metrics = {}
    tensors = {}
    while True:
        try:
            result = sess.run(model.fetches[mode])
            accumulate(metrics, result['metrics'])
            accumulate(tensors, result['tensors'])
        except tf.errors.OutOfRangeError:
            break
    metrics = average(metrics)
    save_metrics(
        train_step,
        metrics,
        os.path.join(save_path, 'results/valid')
    )
    save_tensors(average(tensors), os.path.join(save_path, 'tensors'))
    return metrics


def train(model, config, save_path):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    log_steps = logarithmic_int_range(
        0, config['num_steps'], config['log_factor'], True)
    with tf.Session(config=sess_config) as sess:
        sess.run(
            [
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            ]
        )
        sess.run(model.init_ops['train'])
        for step in range(config['num_steps']+1):
            if step in log_steps:
                valid_metrics = test(
                    sess, step, model, 'valid', save_path)
                sess.run(model.init_ops['train'])
                if step > 0:
                    save_metrics(
                        step,
                        res['metrics'],
                        os.path.join(save_path, 'results/train')
                    )
                log(
                    step,
                    res['metrics'] if step > 0 else None,
                    valid_metrics
                )
            lr = config['lr_init'] \
                * config['lr_decay'] ** (step // config["lr_step"])
            res = sess.run(
                model.fetches['train'],
                feed_dict={model.feed_dict['lr']: lr}
            )


def get_mnist():
    data_dir = os.path.join(get_repo_root('nc-ff'), 'datasets')
    train_ds = tfds.load(
        name="mnist:3.*.*",
        split="train[:80%]",
        batch_size=17,
        as_supervised=True,
        data_dir=data_dir
    )
    valid_ds = tfds.load(
        name="mnist:3.*.*",
        split="train[-20%:]",
        batch_size=400,
        as_supervised=True,
        data_dir=data_dir
    )
    train_ds = train_ds.repeat().shuffle(1024)
    print(type(train_ds))
    return {'train': train_ds, 'valid': valid_ds}


def launch(config):
    model = DenseNetwork(config['graph'])
    datasets = get_mnist()
    model.build(datasets)
    train(model, config['train'], config['save_path'])


def distribute(config):
    for i in range(config['num_repeats']):
        launch_config = copy.deepcopy(config)
        launch_config['save_path'] = os.path.join(
            launch_config['save_path'], str(i))
        p = mp.Process(target=launch, args=(launch_config,))
        p.start()
        p.join()


def main():
    args = get_args()
    with open(args.config) as f:
        config = json.load(f)
    save_path = get_save_path(args.config, 'nc-ff', 'configs', 'results')
    get_mnist()
    config['save_path'] = save_path
    distribute(config)


if __name__ == '__main__':
    main()
