import argparse
import copy
import datetime
import json
import multiprocessing as mp
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

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


def log(step, tr_metrics, vd_metrics, lr):
    tmpl = "{time} STEP: {step} || learning rate: {lr} || {valid}"
    msg = tmpl.format(
        time=datetime.datetime.now(), step=step, valid=vd_metrics, lr=lr)
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


def save_metrics_and_params(step, metrics, params, save_path):
    os.makedirs(save_path, exist_ok=True)
    for k, v in metrics.items():
        fn = os.path.join(save_path, k + '.txt')
        with open(fn, 'a') as f:
            if step is None:
                f.write("{}\n".format(v))
            else:
                f.write("{} {}\n".format(step, v))
    for k, v in params.items():
        fn = os.path.join(save_path, k + '.txt')
        with open(fn, 'a') as f:
            if step is None:
                f.write("{}\n".format(v))
            else:
                f.write("{} {}\n".format(step, v))


def save_tensors(tensors, save_path, tensors_and_accumulators_to_save=None):
    if tensors_and_accumulators_to_save is None:
        tensors_and_accumulators_to_save = set()
    os.makedirs(save_path, exist_ok=True)
    for k in set(tensors_and_accumulators_to_save) & set(tensors.keys()):
        v = tensors[k]
        fn = os.path.join(save_path, k + '.pickle')
        with open(fn, 'ab') as f:
            pickle.dump(v, f)


def test(
        sess,
        train_step,
        model,
        mode,
        save_path,
        tensors_and_accumulators_to_save
):
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
    save_metrics_and_params(
        train_step,
        metrics,
        {},
        os.path.join(save_path, 'results/valid')
    )
    save_tensors(
        average(accumulators),
        os.path.join(save_path, 'tensors'),
        tensors_and_accumulators_to_save
    )
    return metrics


def get_lr_decay_method(config):
    if 'lr_step' in config:
        method_of_lr_decay = 'periodic'
    elif 'lr_patience' in config and 'lr_patience_period' in config:
        method_of_lr_decay = 'impatience'
    else:
        raise ValueError(
            "Train config has to contain either parameter 'lr_step' or "
            "parameters 'lr_patience' and 'lr_patience_period'.\n"
            "train config={}".format(config)
        )
    return method_of_lr_decay


def update_lr(lr, step, valid_metrics, best_ce_loss, lr_impatience, config):
    method_of_lr_decay = get_lr_decay_method(config)
    if method_of_lr_decay == 'periodic':
        lr = config['lr_init'] \
             * config['lr_decay'] ** (step // config["lr_step"])
    elif method_of_lr_decay == 'impatience':
        if step % config['lr_patience_period'] == 0:
            if valid_metrics['ce_loss'] < best_ce_loss:
                lr_impatience = 0
                best_ce_loss = valid_metrics['ce_loss']
            else:
                lr_impatience += 1
                if lr_impatience > config['lr_patience']:
                    lr *= config['lr_decay']
                    lr_impatience = 0
    return lr, lr_impatience, best_ce_loss


def get_training_interruption_method(config):
    if 'num_steps' in config:
        method_of_interrupting_training = 'fixed_num_steps'
    elif 'stop_patience_period' in config and 'stop_patience' in config:
        method_of_interrupting_training = 'impatience'
    else:
        raise ValueError(
            "Train config has to contain either parameter 'fixed_num_steps'"
            "or parameters 'stop_patience' and 'stop_patience_period'.\n"
            "train config={}".format(config)
        )
    return method_of_interrupting_training


def decide_if_training_is_finished(
        step, valid_metrics, best_ce_loss, stop_impatience, config):
    method_of_interruption_of_training = get_training_interruption_method(
        config)
    if method_of_interruption_of_training == 'fixed_num_steps':
        stop_training = step > config['num_steps']
    elif method_of_interruption_of_training == 'impatience':
        if step % config['stop_patience_period'] == 0:
            if valid_metrics['ce_loss'] < best_ce_loss:
                stop_impatience = 0
                best_ce_loss = valid_metrics['ce_loss']
            else:
                stop_impatience += 1
        stop_training = stop_impatience > config['stop_patience']
    else:
        raise ValueError(
            "Unsupported method of interrupting training '{}'".format(
                method_of_interruption_of_training
            )
        )
    return stop_training, stop_impatience, best_ce_loss


def time_for_logarithmic_logging(step, factor):
    if step == 0:
        return True
    step_is_integer_power_of_factor = int(np.log(step+1) / np.log(factor)) \
        - int(np.log(step) / np.log(factor)) > 0
    return step_is_integer_power_of_factor


def train(model, config, save_path):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # log_steps = logarithmic_int_range(
    #     0, config['num_steps'], config['log_factor'], True)
    with tf.Session(config=sess_config) as sess:
        sess.run(
            [
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            ]
        )
        sess.run(model.init_ops['train'])

        step = 0
        lr = config['lr_init']

        stop_impatience = 0
        # `lr_impatience` is not used if 'lr_step' is in `config`.
        lr_impatience = 0
        best_stop_ce_loss = float('+inf')
        best_lr_ce_loss = float('+inf')

        while True:
            if time_for_logarithmic_logging(step, config['log_factor']):
                tensors = sess.run(model.fetches['tensors'])
                if 'eigen' in config:
                    tensors = count_real_eigen_values_fraction(tensors)
                save_tensors(
                    tensors,
                    os.path.join(save_path, 'tensors'),
                    config.get('tensors_and_accumulators_to_save')
                )
                valid_metrics = test(
                    sess,
                    step,
                    model,
                    'valid',
                    save_path,
                    config.get('tensors_and_accumulators_to_save')
                )
                sess.run(model.init_ops['train'])
                if step > 0:
                    save_metrics_and_params(
                        step,
                        res['metrics'],
                        {'lr': lr},
                        os.path.join(save_path, 'results/train')
                    )
                log(
                    step,
                    res['metrics'] if step > 0 else None,
                    valid_metrics,
                    lr
                )

            res = sess.run(
                model.fetches['train'], feed_dict={model.feed_dict['lr']: lr})
            step += 1

            lr, lr_impatience, best_lr_ce_loss = update_lr(
                lr,
                step,
                valid_metrics,
                best_lr_ce_loss,
                lr_impatience,
                config
            )

            stop_training, stop_impatience, best_stop_ce_loss = \
                decide_if_training_is_finished(
                    step,
                    valid_metrics,
                    best_stop_ce_loss,
                    stop_impatience,
                    config)
            if stop_training:
                break


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


def sample_by_classes(examples, labels, balance, class_starting_indices):
    class_fractions = balance / np.max(balance)
    class_sizes = [
        int((end-start)*c) for start, end, c in zip(
            class_starting_indices,
            class_starting_indices[1:] + [len(class_starting_indices)],
            class_fractions
        )
    ]
    new_examples, new_labels = [], []
    for start, size in zip(class_starting_indices, class_sizes):
        new_examples.append(examples[start:start+size])
        new_labels.append(labels[start:start+size])
    return np.concatenate(new_examples, axis=0), \
        np.concatenate(new_labels, axis=0)


def get_unbalanced_mnist(
        train_bs,
        valid_bs,
        balance=(.1, .1, .1, .1, .1, .1, .1, .1, .1, .1),
        shuffle_balance=True,
):
    valid_frac = 0.2
    data_dir = os.path.join(get_repo_root('nc-ff'), 'datasets')
    data_file = os.path.join(data_dir, 'mnist_sorted.npz')
    with np.load(data_file) as data:
        train_examples = data['x_train'].reshape((-1, 28, 28, 1)).astype(np.int64)
        train_labels = data['y_train']
        test_examples = data['x_test'].reshape((-1, 28, 28, 1)).astype(np.int64)
        test_labels = data['y_test']
    balance = np.array(balance)
    if shuffle_balance:
        np.random.shuffle(balance)
    train_examples, train_labels = sample_by_classes(
        train_examples,
        train_labels,
        balance=balance,
        class_starting_indices=[len(train_labels) // 10 * i for i in range(10)]
    )
    train_examples, valid_examples, train_labels, valid_labels = \
        train_test_split(
            train_examples,
            train_labels,
            test_size=valid_frac,
            stratify=train_labels
        )
    test_examples, test_labels = sample_by_classes(
        test_examples,
        test_labels,
        balance,
        class_starting_indices=[len(test_labels) // 10 * i for i in range(10)]
    )
    sizes = {
        'train': len(train_labels),
        'valid': len(valid_labels),
        'test': len(test_labels)
    }
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_examples, train_labels))
    valid_ds = tf.data.Dataset.from_tensor_slices(
        (valid_examples, valid_labels))
    valid_ds = valid_ds.batch(valid_bs)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_examples, test_labels))
    test_ds = test_ds.batch(sizes['test'])
    train_ds = train_ds.repeat().shuffle(1024).batch(train_bs)
    return {'train': train_ds, 'valid': valid_ds, "test": test_ds}, sizes


def get_datasets(config):
    dataset_config = config['train'].get('dataset', {'name': 'mnist'})
    if dataset_config['name'] == 'mnist':
        datasets, sizes = get_mnist(
            config['train']['batch_size'],
            config['train']['valid']['batch_size'])
    elif dataset_config['name'] == 'unbalanced_mnist':
        print('Unbalanced dataset')
        datasets, sizes = get_unbalanced_mnist(
            config['train']['batch_size'],
            config['train']['valid']['batch_size'],
            balance=config['train']['dataset']['balance'],
        )
    else:
        raise ValueError("Unsupported dataset type.")
    return datasets, sizes


def launch(config):
    ModelClass = getattr(nets, config['graph']['net'])
    model = ModelClass(config['graph'])
    datasets, sizes = get_datasets(config)
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
