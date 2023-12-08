"""
Trainer for Conditional Normalizing Flow
"""
import os
import time
import re

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils import train_density_estimation_cond
from utils_plot import *

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

    
def evaluate(flow_model, testing_data, cond_kwargs):
    num_samples, num_dims = testing_data.shape
    samples = flow_model.sample(num_samples, bijector_kwargs=cond_kwargs).numpy()
    
    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
            for idx in range(num_dims)
    ]
    return np.average(distances), samples


def train(train_truth, train_in, testing_truth, test_in, flow_model, lr, batch_size, layers, max_epochs, outdir, plot_config):
    base_lr = lr
    end_lr = 1e-5
    nbatches = train_truth.shape[0] // batch_size + 1
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        base_lr, max_epochs * nbatches, end_lr, power=0.5)

    # initialize checkpoints
    checkpoint_directory = "{}/checkpoints".format(outdir)
    os.makedirs(checkpoint_directory, exist_ok=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=flow_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=None)
    latest_ckpt = ckpt_manager.latest_checkpoint
    _ = checkpoint.restore(latest_ckpt).expect_partial()
    
    print("Loading latest checkpoint from: {}".format(checkpoint_directory))
    if latest_ckpt:
        start_epoch = int(re.findall(r'\/ckpt-(.*)', latest_ckpt)[0]) + 1
        print("Restored from {}".format(latest_ckpt))
    else:
        start_epoch = 0
        print("Initializing from scratch.")
    
    labels = plot_config['labels']
    nphotons = plot_config['nphotons']
    cond_keep = plot_config['cond_keep']
    
    train_in = train_in[:, cond_keep]
    test_in_plot = test_in # include truth phi for plotting
    test_in = test_in[:, cond_keep]
    
    AUTO = tf.data.experimental.AUTOTUNE
    training_data = tf.data.Dataset.from_tensor_slices(
        (train_in, train_truth)).batch(batch_size).prefetch(AUTO)

    img_dir = os.path.join(outdir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    log_dir = os.path.join(outdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_dir = os.path.join(outdir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # start training
    summary_logfile = os.path.join(log_dir, 'results.txt')
    tmp_res = "# Epoch, Time, WD (Wasserstein distance), Ltr (training loss), Lte (testing loss)" 
    with open(summary_logfile, 'a') as f:
        f.write(tmp_res + "\n")
    # save learning rate
    lr_logfile = os.path.join(log_dir, 'learning_rate.txt')
    tmp_res = "# Epoch, " + ", ".join(map(str, range(nbatches)))
    with open(lr_logfile, 'a') as f:
        f.write(tmp_res + "\n")
    print("idx, train loss, distance, minimum distance, minimum epoch")
    
    # Get min wass dist so far
    df = pd.read_csv(summary_logfile)
    df.columns = ['epoch', 'time', 'wass_dist', 'loss_tr', 'loss_te']
    df = df[df['epoch'].str.contains(r'\*')]
    if len(df) == 0:
        min_wdis, min_iepoch = 9999, -1
    else:
        df['epoch'] = df['epoch'].apply(lambda x: x[2:])
        df = df.reset_index(drop = True).astype('float64')
        min_wdis, min_iepoch = df[['wass_dist', 'epoch']].sort_values('wass_dist').iloc[0]
        min_iepoch = int(min_iepoch)
    
    delta_stop = 1000
    start_time = time.time()
    testing_batch = tf.cast(tf.convert_to_tensor(testing_truth), 'float32')
    testing_cond = tf.cast(tf.convert_to_tensor(test_in), 'float32')
    cond_kwargs = dict([(f"b{idx}", {"conditional_input": testing_cond}) for idx in range(layers)])

    for i in range(start_epoch, max_epochs):
        lr = []
        train_loss = []
        for batch in training_data:
            condition, batch = batch
            batch = tf.cast(batch, 'float32')
            condition = tf.cast(condition, 'float32')
            train_loss += [train_density_estimation_cond(flow_model, opt, batch, condition, layers)]
            #lr += [opt.lr.numpy()]
            lr += [opt.lr(opt.iterations).numpy()]
            
        train_loss = np.array(train_loss)
        avg_loss = np.sum(train_loss, axis=0)/train_loss.shape[0]
        elapsed = time.time() - start_time
        
        wdis, predictions = evaluate(flow_model, testing_truth, cond_kwargs)
        test_loss = -tf.reduce_mean(flow_model.log_prob(testing_batch, bijector_kwargs=cond_kwargs))
        
        if wdis < min_wdis:
            min_wdis = wdis
            min_iepoch = i
            compare(predictions, testing_truth, img_dir, i, [-1, 1], labels, nphotons)
            plot_corr(predictions, img_dir, i, labels)
            if nphotons == 2:
                plot_yy_cond(predictions, testing_truth, test_in_plot, 
                             labels, plot_config['scale'], img_dir, i, nphotons)
            
            save_path = os.path.join(csv_dir, 'pred_epoch_{:04d}.csv'.format(i))
            pd.DataFrame(predictions, columns = labels).to_csv(save_path, index=False)
        elif i - min_iepoch > delta_stop:
            plot_logfile(summary_logfile, i, img_dir, lr_logfile)
            break
        ckpt_manager.save(checkpoint_number = i)
        
        if (i % 50 == 0) or (i == max_epochs - 1):
            plot_logfile(summary_logfile, i, img_dir, lr_logfile)
        if i == 0:
            plot_corr(testing_truth, img_dir, i, labels, save_path = os.path.join(img_dir, 'corr_epoch_{:04d}_truth.png'.format(i)))

        tmp_res = "* {:05d}, {:.1f}, {:.4f}, {:.4f}, {:.4f}".format(i, elapsed, wdis, avg_loss, test_loss)
        with open(summary_logfile, 'a') as f:
            f.write(tmp_res + "\n")
        tmp_res = ", ".join(map(str, lr))
        with open(lr_logfile, 'a') as f:
            f.write("* {:05d}, ".format(i) + tmp_res + "\n")
        print(f"{i}, {train_loss[-1]:.4f}, {wdis:.4f}, {min_wdis:.4f}, {min_iepoch}", flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg("--config_file", help='configuration file')
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--epochs", help='number of maximum epochs', default=500, type=int)
    add_arg("--max-evts", help="maximum number of events", default=None, type=int)
    
    args = parser.parse_args()
    
    import preprocess
    from made import create_conditional_flow
    
    # =============================
    # load preprocessed data
    # =============================
    config = preprocess.load_yaml(args.config_file)
    file_name = config['file_name']
    tree_name = config['tree_name']
    out_branch_names = config['out_branch_names']
    truth_branch_names = config['truth_branch_names']
    data_branch_names = config['data_branch_names']
    
    nphotons = config['nphotons']
    # indices of conditions to use
    cond_keep = config['cond_keep']
    
    data, scale, label = preprocess.read_data_root(file_name, tree_name, 
                                                   out_branch_names,
                                                   in_branch_names=truth_branch_names,
                                                   data_branch_names=data_branch_names, 
                                                   max_evts=args.max_evts, 
                                                   num_particles=nphotons,
                                                   use_cond=True,
                                                   sample=config['sample'])
    train_in, train_truth, train_other, val_in, val_truth, val_other, test_in, test_truth, test_other = data
    in_branch_names, out_branch_names, data_branch_names = label
    
    print('\n{} output, {} input, {} total data'.format(len(out_branch_names), len(cond_keep), len(data_branch_names)))
    print('Generating:', out_branch_names)
    print('Input data:', [b for i, b in enumerate(in_branch_names) if i in cond_keep], '\n')
    
    print('training:', train_truth.shape, 
          'validation:', val_truth.shape,
          'test:', test_truth.shape)
    
    outdir = os.path.join('trained_results', args.log_dir)
    hidden_shape = [config['latent_size']]*config['num_layers']
    layers = config['num_bijectors']
    activation = config['activation']
    lr = config['lr']
    batch_size = config['batch_size']
    max_epochs = args.epochs
    
    dim_truth = train_truth.shape[1]
    dim_cond = len(cond_keep)
    maf = create_conditional_flow(hidden_shape, layers, 
                                  input_dim=dim_truth, conditional_event_shape=(dim_cond,), out_dim=2, 
                                  activation=activation)
    
    plot_config = {'nphotons': nphotons,
                   'labels': out_branch_names,
                   'scale': scale,
                   'cond_keep': cond_keep}
    train(train_truth, train_in, val_truth, val_in, maf, lr, batch_size, layers, max_epochs, outdir, plot_config)
