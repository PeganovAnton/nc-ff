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

import nets


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
    accumulators = {}
    while True:
        try:
            result = sess.run(model.fetches[mode])
            accumulate(metrics, result['metrics'])
            accumulate(accumulators, result['accumulators'])
        except tf.errors.OutOfRangeError:
            break
    metrics = average(metrics)
    save_metrics(
        train_step,
        metrics,
        os.path.join(save_path, 'results/valid')
    )
    save_tensors(average(accumulators), os.path.join(save_path, 'tensors'))
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
                tensors = sess.run(model.fetches['tensors'])
                tensors = count_real_eigen_values_fraction(tensors)
                save_tensors(tensors, os.path.join(save_path, 'tensors'))
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


def get_mnist(train_bs, valid_bs, test_bs=10000):
    split_percentage = 80
    data_dir = os.path.join(get_repo_root('nc-ff'), 'datasets')
    train_ds, info = tfds.load(
        name="mnist:3.*.*",
        split="train[:{}%]".format(split_percentage),
        batch_size=train_bs,
        as_supervised=True,
        data_dir=data_dir,
        with_info=True
    )
    valid_ds = tfds.load(
        name="mnist:3.*.*",
        split="train[-{}%:]".format(100 - split_percentage),
        batch_size=valid_bs,
        as_supervised=True,
        data_dir=data_dir
    )
    test_ds = tfds.load(
        name="mnist:3.*.*",
        split="test",
        batch_size=test_bs,
        as_supervised=True,
        data_dir=data_dir
    )
    splits = info.splits
    frac = split_percentage / 100
    sizes = {
        'train': round(splits['train'].num_examples * frac),
        'valid':
            splits['train'].num_examples
            - round(splits['train'].num_examples * frac),
        'test': splits['test'].num_examples
    }
    train_ds = train_ds.map(
        lambda inp, lbl: (tf.cast(inp, dtype=tf.int64), lbl))
    valid_ds = valid_ds.map(
        lambda inp, lbl: (tf.cast(inp, dtype=tf.int64), lbl))
    test_ds = test_ds.map(
        lambda inp, lbl: (tf.cast(inp, dtype=tf.int64), lbl))
    train_ds = train_ds.repeat().shuffle(1024)
    return {'train': train_ds, 'valid': valid_ds, "test": test_ds}, sizes


def launch(config):
    ModelClass = getattr(nets, config['graph']['net'])
    model = ModelClass(config['graph'])
    datasets, sizes = get_mnist(
        config['train']['batch_size'],
        config['train']['valid']['batch_size'])
    model.build(datasets, sizes)
    train(model, config['train'], config['save_path'])


def distribute(config):
    for i in range(config['num_repeats']):
        print('\nLAUNCH #{}'.format(i))
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
    # p = mp.Process(target=get_mnist, args=())
    # p.start()
    # p.join()
    save_path = get_save_path(args.config, 'nc-ff', 'configs', 'results')
    config['save_path'] = save_path
    distribute(config)


def count_real_eigen_values_fraction(tensors):
    for k in list(filter(lambda x: x[:6] == 'kernel', tensors.keys())):
        v = tensors[k]
        sh = v.shape
        if sh[0] == sh[1]:
            e, v = np.linalg.eig(v)
            n = np.sum(np.iscomplex(e).astype(int))
            d = v.shape[0]
            tensors[k] = (d - n) / d
        else:
            del tensors[k]
    return tensors


if __name__ == '__main__':
    main()
