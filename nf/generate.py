import os
import time
import re
import h5py

import tensorflow as tf 
import numpy as np
import pandas as pd
from scipy import stats

import preprocess
from made import create_conditional_flow

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

import argparse
parser = argparse.ArgumentParser(description='Generate')
add_arg = parser.add_argument
add_arg("--config_file", help='configuration file')
add_arg("--log-dir", help='log directory', default='log_training')
add_arg("--epochs-total", help='total number of epochs', default=500, type=int)
add_arg("--epoch", help='epoch used to generate', default=-1, type=int) # default: epoch with min WD on validation set
add_arg("--dataset", help='train/val/test', default='test')
add_arg("--max-evts", help="maximum number of events", default=None, type=int)

args = parser.parse_args()
config_file = args.config_file
model_name = args.log_dir
max_epochs = args.epochs_total
max_evts = args.max_evts
dataset = args.dataset
best_epoch = args.epoch

outdir = os.path.join('trained_results', model_name)
log_dir = os.path.join(outdir, 'logs')
csv_dir = os.path.join(outdir, 'csv')
summary_logfile = os.path.join(log_dir, 'results.txt')

if best_epoch == -1:
    # Get epoch with min wass dist on validation set
    df = pd.read_csv(summary_logfile)
    df.columns = ['epoch', 'time', 'wass_dist', 'loss_tr', 'loss_te']
    df = df[df['epoch'].str.contains(r'\*')]
    df['epoch'] = df['epoch'].apply(lambda x: x[2:])
    df = df.reset_index(drop = True).astype('float64')
    min_wdis, min_iepoch = df[['wass_dist', 'epoch']].sort_values('wass_dist').iloc[0]
    best_epoch = int(min_iepoch)

# =============================
# load preprocessed data
# =============================
config = preprocess.load_yaml(config_file)
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
                                               max_evts=max_evts, 
                                               num_particles=nphotons,
                                               use_cond=True,
                                               sample=config['sample'])

train_in, train_truth, train_other, val_in, val_truth, val_other, test_in, test_truth, test_other = data
labels_in, labels, labels_other = label
branch_scale, truth_in_scale, input_data_scale = scale

if dataset == 'train':
    test_in, test_truth, test_other = train_in, train_truth, train_other
elif dataset == 'val':
    test_in, test_truth, test_other = val_in, val_truth, val_other

# load model
hidden_shape = [config['latent_size']]*config['num_layers']
layers = config['num_bijectors']
activation = config['activation']
lr = config['lr']
batch_size = config['batch_size']

dim_truth = train_truth.shape[1]
dim_cond = len(cond_keep)
flow_model = create_conditional_flow(hidden_shape, layers, 
                                     input_dim=dim_truth, conditional_event_shape=(dim_cond,), out_dim=2,
                                     activation=activation)


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
latest_ckpt = checkpoint_directory+'/ckpt-'+str(best_epoch)
_ = checkpoint.restore(latest_ckpt).expect_partial()

print("Loading latest checkpoint from: {}".format(checkpoint_directory))
if latest_ckpt:
    start_epoch = int(re.findall(r'\/ckpt-(.*)', latest_ckpt)[0]) + 1
    print("Restored from {}".format(latest_ckpt))
else:
    start_epoch = 0
    print("Initializing from scratch.")
    

# generate
def evaluate(flow_model, testing_data, cond_kwargs):
    num_samples, num_dims = testing_data.shape
    samples = flow_model.sample(num_samples, bijector_kwargs=cond_kwargs).numpy()
    
    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
            for idx in range(num_dims)
    ]
    return np.average(distances), samples

testing_batch = tf.cast(tf.convert_to_tensor(test_truth), 'float32')
testing_cond = tf.cast(tf.convert_to_tensor(test_in[:, cond_keep]), 'float32')
cond_kwargs = dict([(f"b{idx}", {"conditional_input": testing_cond}) for idx in range(layers)])

wdis, predictions = evaluate(flow_model, test_truth, cond_kwargs)
print('Model {}, epoch {}, dataset {}'.format(model_name, best_epoch, dataset))
print('Wasserstein distance', round(wdis, 4))

# save csv with generation
save_path = os.path.join(csv_dir, 'pred_epoch_{:04d}_{}.csv'.format(best_epoch, dataset))
predictions = pd.DataFrame(predictions, columns = labels)
predictions.to_csv(save_path, index=False)

# save h5
h5_dir = os.path.join(outdir, 'h5')
os.makedirs(h5_dir, exist_ok=True)
file = os.path.join(h5_dir, '{}{:04d}.hdf5'.format(dataset, best_epoch))
print('Saving', file)

f = h5py.File(file, "w")
f.create_dataset('scale', data = truth_in_scale)
f.create_dataset('condition_scale', data = input_data_scale)
f.create_dataset('condition_keep', data = np.array(cond_keep))
f.close()

df_test_in = pd.DataFrame(test_in, columns = labels_in)
df_test_truth = pd.DataFrame(test_truth, columns = labels)
pred_scaled = pd.DataFrame(predictions, columns = labels) * truth_in_scale
truths_scaled = pd.DataFrame(test_truth, columns = labels) * truth_in_scale

test_in_scaled = test_in * input_data_scale
pred_feat = test_in_scaled[:,:dim_truth] + pred_scaled
truth_feat = test_in_scaled[:,:dim_truth] + truths_scaled

labels_feat = ['ph_pt', 'ph_phi', 'ph_eta']
if nphotons > 1:
    labels_feat = [label+str(i) for i in range(1, nphotons+1) for label in labels_feat]
pred_feat.columns = labels_feat
truth_feat.columns = labels_feat

# normalized
df_test_in.to_hdf(file, 'condition', mode='a', format='table')
predictions.to_hdf(file, 'generated', mode='a', format='table')
df_test_truth.to_hdf(file, 'simulated', mode='a', format='table')

# reco-level
pred_feat.to_hdf(file, 'generated_reco', mode='a', format='table')
truth_feat.to_hdf(file, 'simulated_reco', mode='a', format='table')