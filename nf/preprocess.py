import vector
import uproot
import yaml
import pandas as pd
import numpy as np
import re
import itertools
import configparser


def load_yaml(filename):
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def read_data_root(file_name, tree_name,
                   out_branch_names, # variables to generate
                   in_branch_names=None, # optional: conditional variables to input
                   data_branch_names=None, # optional: additional variables
                   max_evts=None, test_frac=0.1, val_frac=0.1,
                   fixed_num_particles=True, num_particles=2, 
                   use_cond=False, sample='delphes'):
    """
    Read ROOT data file
    Given filename or list of filenames, return train/test inputs and truth
    """
    if type(file_name) != list:
        file_name = [file_name]
    if not in_branch_names:
        in_branch_names = []
    if not data_branch_names:
        data_branch_names = []
    
    dfs = []
    branch_names = in_branch_names + out_branch_names + data_branch_names
    for f in file_name:
        in_file = uproot.open(f)
        tree = in_file[tree_name]
        array_root = tree.arrays(branch_names, library="np")
        df_root = pd.DataFrame(array_root)
        dfs += [df_root]
    df = pd.concat(dfs)
    data_branch_names = branch_names

    if use_cond:
        # get truth variables
        if sample == 'ggH':
            df = df[df['isPassed']].reset_index(drop=True)
            
        if sample == 'delphes':
            df['fold'] = (df['GenMET_phi'] * 1000 % 10).astype(int)
            for i in [0, 1]:
                df['ph_pt'+str(i+1)] = df['smearph_pt'].apply(lambda x: x[i])
                df['ph_phi'+str(i+1)] = df['smearph_phi'].apply(lambda x: x[i])
                df['ph_eta'+str(i+1)] = df['smearph_eta'].apply(lambda x: x[i])

                df['ph_truth_pt'+str(i+1)] = df['genph_pt'].apply(lambda x: x[i])
                df['ph_truth_phi'+str(i+1)] = df['genph_phi'].apply(lambda x: x[i])
                df['ph_truth_eta'+str(i+1)] = df['genph_eta'].apply(lambda x: x[i])
            cols_ordered = ['ph_truth_pt1', 'ph_truth_phi1', 'ph_truth_eta1', 'ph_truth_pt2', 'ph_truth_phi2', 'ph_truth_eta2']
            data_branch_names += cols_ordered + ['fold']
        else:
            df, cols_ordered, cols_added = get_truth_ph(df, sample)
            data_branch_names += cols_added
    
        # get detector response
        in_branch_names = cols_ordered + ['mu']
        out_branch_names = ['ph_diff_pt1', 'ph_diff_phi1', 'ph_diff_eta1', 'ph_diff_pt2', 'ph_diff_phi2', 'ph_diff_eta2']
        
        df['ph_diff_pt1'] = df['ph_pt1'] - df['ph_truth_pt1']
        df['ph_diff_phi1'] = df['ph_phi1'] - df['ph_truth_phi1']
        df['ph_diff_eta1'] = df['ph_eta1'] - df['ph_truth_eta1']
        df['ph_diff_pt2'] = df['ph_pt2'] - df['ph_truth_pt2']
        df['ph_diff_phi2'] = df['ph_phi2'] - df['ph_truth_phi2']
        df['ph_diff_eta2'] = df['ph_eta2'] - df['ph_truth_eta2']
        
        x = df['ph_diff_phi1'] > np.pi
        df.loc[x, 'ph_diff_phi1'] = df.loc[x, 'ph_diff_phi1'] - 2*np.pi
        x = df['ph_diff_phi1'] < -np.pi
        df.loc[x, 'ph_diff_phi1'] = df.loc[x, 'ph_diff_phi1'] + 2*np.pi

        x = df['ph_diff_phi2'] > np.pi
        df.loc[x, 'ph_diff_phi2'] = df.loc[x, 'ph_diff_phi2'] - 2*np.pi
        x = df['ph_diff_phi2'] < -np.pi
        df.loc[x, 'ph_diff_phi2'] = df.loc[x, 'ph_diff_phi2'] + 2*np.pi

        if num_particles == 2:
            drop_events = (np.abs(df['ph_diff_pt1']) > 20) | (np.abs(df['ph_diff_pt2']) > 20)
            df = df[~drop_events]
        elif num_particles == 1:
            # reshape to one particle
            cols_ph1 = [b for b in df.columns if '1' in b]
            cols_ph2 = [b for b in df.columns if '2' in b]
            cols = [b[:-1] for b in cols_ph1]
            
            df_ph1 = df[cols_ph1]
            df_ph2 = df[cols_ph2]
            df_ph1.columns = cols
            df_ph2.columns = cols
            df = pd.concat([df_ph1, df_ph2])
            
            drop_events = (np.abs(df['ph_diff_pt']) > 20)
            df = df[~drop_events]

            data_branch_names = [b[:-1] for b in data_branch_names if '1' in b]
            in_branch_names = [b[:-1] for b in in_branch_names if '1' in b]
            out_branch_names = [b[:-1] for b in out_branch_names if '1' in b]
    
    print('\n{} output, {} input, {} total data'.format(len(out_branch_names), len(in_branch_names), len(data_branch_names)))
    print('Generating:', out_branch_names)
    print('Input data:', in_branch_names, '\n')
    
    out_particles = df[out_branch_names]
    branch_scale = {}
    
    # transform pt
    if not use_cond:
        for b in out_branch_names:
            if 'pt' in b:
                x = np.log(out_particles[b])
                branch_scale[b] = np.median(x)
                x = x - np.median(x)
                out_particles.loc[:, b] = x
    print("Branch scale:", branch_scale)
        
    if use_cond:
        truth_in_scale = np.array([20, 0.01, 0.005]) # pt, phi, eta
        if num_particles > 1:
            truth_in_scale = np.array(list(truth_in_scale) * num_particles)
        truth_in = out_particles / truth_in_scale
        truth_in = truth_in.to_numpy()
    else:
        # scale to (-1, 1)
        truth_in_scale = out_particles.abs().max() + 1e-6
        truth_in = out_particles / truth_in_scale
        truth_in = truth_in.to_numpy()
        print("Truth scale:", dict(truth_in_scale))
    
    if len(in_branch_names) > 0:
        input_data = df[in_branch_names]
        if not use_cond:
            for b in in_branch_names:
                if 'pt' in b:
                    x = np.log(input_data[b])
                    x = x - np.median(x)
                    input_data.loc[:, b] = x
        
        input_data_scale = input_data.abs().max()
        input_data = input_data / input_data_scale
        input_data = input_data.to_numpy()
    else:
        input_data = np.array([])
        input_data_scale = np.array([])
    print("Input scale:", dict(input_data_scale))
        
    if len(data_branch_names) > 0:
        other_data = df[data_branch_names].to_numpy()
    else:
        other_data = np.array([])
    
    # for tanh
    drop = (np.abs(truth_in).max(axis=1) >= 1)
    truth_in = truth_in[~drop]
    input_data = input_data[~drop]
    other_data = other_data[~drop]
    
    if sample == 'delphes':
        # Split by fold
        fold = df[~drop]['fold'].values      
        test = (fold == 0)
        val = (fold == 1)
        train = (fold >= 2)
        test_in, val_in, train_in = input_data[test], input_data[val], input_data[train]
        test_truth, val_truth, train_truth = truth_in[test], truth_in[val], truth_in[train]
        test_other, val_other, train_other = other_data[test], other_data[val], other_data[train]
    else:
        # Split by random shuffling
        num_evts = truth_in.shape[0]
        if max_evts and (truth_in.shape[0] > max_evts):
            num_evts = max_evts
        num_test_evts = int(num_evts*test_frac)
        num_val_evts = int(num_evts*val_frac) + num_test_evts

        # <NOTE, https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html>
        from numpy.random import MT19937
        from numpy.random import RandomState, SeedSequence
        np_rs = RandomState(MT19937(SeedSequence(123456789)))
        idx = np.arange(df.shape[0])
        np_rs.shuffle(idx)
        if len(input_data) > 0:
            input_data = input_data[idx]
        if len(truth_in) > 0:
            truth_in = truth_in[idx]
        if len(other_data) > 0:
            other_data = other_data[idx]

        # empty arrays for test_in and train_in if no conditions
        test_in, val_in, train_in = input_data[:num_test_evts], input_data[num_test_evts:num_val_evts], input_data[num_val_evts:num_evts]
        test_truth, val_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:num_val_evts], truth_in[num_val_evts:num_evts]
        test_other, val_other, train_other = other_data[:num_test_evts], other_data[num_test_evts:num_val_evts], other_data[num_val_evts:num_evts]

    if len(input_data) == 0:
        train_in = None
        val_in = None
        test_in = None

    data = (train_in, train_truth, train_other, val_in, val_truth, val_other, test_in, test_truth, test_other)
    scale = (branch_scale, truth_in_scale, input_data_scale)
    label = (in_branch_names, out_branch_names, data_branch_names)
        
    print('training:', train_truth.shape, 
          'validation:', val_truth.shape,
          'test:', test_truth.shape)
    
    to_return = (data, scale, label)
    return to_return


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi
    return dphi


def get_truth_ph(df, sample):
    def get_ph_idx_ttH(x):
        pid = np.where(x['m_stable_PID'] == 22)[0]
        pid = pid[np.abs(x['m_stable_eta'][pid]) <= 2.5]
        eid = np.where((x['m_stable_PID'] == 11) | (x['m_stable_PID'] == -11))[0]
        pid_ret = pid[:2]
        if len(pid) > 0:
            p = pid[0]
            eid = eid[eid < p]
            for e in eid:
                # check dR
                delta_eta = x['m_stable_eta'][e] - x['m_stable_eta'][p]
                delta_phi = calc_dphi(x['m_stable_phi'][e], x['m_stable_phi'][p])
                delta_r = np.sqrt(delta_eta ** 2 + delta_phi ** 2)
                if delta_r < 0.1:
                    pid_ret = pid[1:3]
                    break
        if len(pid_ret) < 2:
            pid_ret = np.concatenate([pid_ret, np.repeat(-1, 2-len(pid_ret))])
        else:
            pid_sort = np.argsort(-x['m_stable_pt'][pid_ret])
            pid_ret = pid_ret[pid_sort]
        return pid_ret
    
    def get_ph_idx_ggH(x):
        num_ph = len(x['m_stable_pt'])
        if num_ph == 0:
            pid_ret = [-1, -1]
        elif num_ph == 1:
            pid_ret = [0, -1]
        else:
            ph_pairs = list(itertools.permutations(range(num_ph), 2))
            dR = []
            dR_pairs = []
            for ph1, ph2 in ph_pairs:
                delta_eta1 = x['ph_eta1'] - x['m_stable_eta'][ph1]
                delta_phi1 = calc_dphi(x['ph_phi1'], x['m_stable_phi'][ph1])
                delta_r1 = np.sqrt(delta_eta1 ** 2 + delta_phi1 ** 2)
                if delta_r1 >= 0.2:
                    continue

                delta_eta2 = x['ph_eta2'] - x['m_stable_eta'][ph2]
                delta_phi2 = calc_dphi(x['ph_phi2'], x['m_stable_phi'][ph2])
                delta_r2 = np.sqrt(delta_eta2 ** 2 + delta_phi2 ** 2)
                if delta_r2 >= 0.2:
                    continue

                dR.append(delta_r1 + delta_r2)
                dR_pairs.append([ph1, ph2])

            if len(dR) == 0:
                pid_ret = [-1, -1]
            else:
                dR_min = np.argmin(dR)
                pid_ret = dR_pairs[dR_min]
        return pid_ret

    if sample == 'ttH':
        x = df.apply(get_ph_idx_ttH, axis = 1)
    elif sample == 'ggH':
        x = df.apply(get_ph_idx_ggH, axis = 1)
    
    ph = pd.DataFrame(x.tolist(), columns = ['ph_truth_idx1', 'ph_truth_idx2'], dtype=int)
    df = pd.concat([df, ph], axis=1)
    # drop events if unable to identify 2 truth photons
    no_truth = (df['ph_truth_idx1'] == -1) | (df['ph_truth_idx2'] == -1)
    df = df[~no_truth]
    
    cols_added = []
    for var in ['pt', 'eta', 'phi', 'm']:
        print(var)
        colname = 'm_stable_'+var
        colnew = 'ph_truth_'+var+'1'
        cols_added.append(colnew)
        df[colnew] = df.apply(lambda x: x[colname][x['ph_truth_idx1']], axis = 1)
        colnew = 'ph_truth_'+var+'2'
        cols_added.append(colnew)
        df[colnew] = df.apply(lambda x: x[colname][x['ph_truth_idx2']], axis = 1)
    
    # exclude m, reorder
    order = [0, 4, 2, 1, 5, 3]
    cols_input = list(np.array(cols_added)[order])
    return df, cols_input, cols_added
    