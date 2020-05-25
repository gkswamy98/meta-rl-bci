import numpy as np
from tensorflow.keras import utils as np_utils
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

def denoise(sample, num_channels=8):
    for channel in range(num_channels):
        DataFilter.perform_rolling_filter(sample[0][channel], 3, AggOperations.MEDIAN.value)
        DataFilter.perform_wavelet_denoising(sample[0][channel], 'db6', 3)
    sample = sample / np.expand_dims(sample.std(axis=-1), axis=-1)
    return sample

def pad_sample(sample, sample_len=250):
    if sample.shape[-1] < sample_len:
        sample = np.pad(sample, [(0, 0), (0, 0), (0, sample_len - sample.shape[-1])], mode="mean")
    return sample

def load_data(data_idx, freq=250, num_actions=4):
    # Load Data
    correct_times = np.load("./data/correct_times_{0}.npy".format(data_idx))
    incorrect_times = np.load("./data/incorrect_times_{0}.npy".format(data_idx))
    action_times = [np.load("./data/action_{0}_times_{1}.npy".format(act, data_idx)) for act in range(num_actions)]
    data = np.squeeze(np.load("./data/eeg_data_{0}.npy".format(data_idx)))
    fft_data = np.squeeze(np.load("./data/eeg_fft_data_{0}.npy".format(data_idx)))
    data_ts = np.load("./data/eeg_timestamps_{0}.npy".format(data_idx))
    correct_ts = correct_times.astype(np.float64)
    incorrect_ts = incorrect_times.astype(np.float64)
    action_ts = [x.astype(np.float64) for x in action_times]
    # Order Events (ERP)
    correct_idx = 0
    incorrect_idx = 0
    ts = []
    is_errp = []
    while correct_idx < len(correct_ts) or incorrect_idx < len(incorrect_ts):
        if correct_idx < len(correct_ts) and incorrect_idx >= len(incorrect_ts):
            ts.append(correct_ts[correct_idx])
            is_errp.append(False)
            correct_idx += 1
        elif correct_idx >= len(correct_ts) and incorrect_idx <= len(incorrect_ts):
            ts.append(incorrect_ts[incorrect_idx])
            is_errp.append(True)
            incorrect_idx += 1
        else:
            if correct_ts[correct_idx] < incorrect_ts[incorrect_idx]:
                ts.append(correct_ts[correct_idx])
                is_errp.append(False)
                correct_idx += 1
            else:
                ts.append(incorrect_ts[incorrect_idx])
                is_errp.append(True)
                incorrect_idx += 1
    # Order Events (CTRL)
    action_idxs = [0 for _ in range(num_actions)]
    ordered_labels = []
    ts2 = []
    done = all([action_idxs[i] >= len(action_ts[i]) for i in range(num_actions)])
    while not done:
        valid = [i for i in range(num_actions) if action_idxs[i] < len(action_ts[i])]
        times = [action_ts[i][action_idxs[i]] for i in valid]
        min_idx = np.argmin(times, axis=-1)
        ts2.append(times[min_idx])
        ordered_labels.append(valid[min_idx])
        action_idxs[valid[min_idx]] += 1
        done = all([action_idxs[i] >= len(action_ts[i]) for i in range(num_actions)])
    # Create epochs (ERP)
    error_p = [[]]
    nerror_p = [[]]
    ts_idx = 0
    bad_idx = []
    for i in range(len(data)):
        t = data_ts[i]
        delta = (t - ts[ts_idx])
        if delta < 0:
            pass
        elif delta < 2.5 and delta > 0.:
            if is_errp[ts_idx]:
                error_p[-1].append(data[i])
            else:
                nerror_p[-1].append(data[i])
        elif delta > 2.5:
            if is_errp[ts_idx]:
                if len(error_p[-1]) < freq:
                    error_p = error_p[:-1]
                    bad_idx.append(ts_idx)
            else:
                if len(nerror_p[-1]) < freq:
                    nerror_p = nerror_p[:-1]
                    bad_idx.append(ts_idx)
            ts_idx += 1
            error_p.append([])
            nerror_p.append([])
            if ts_idx == len(ts):
                break
    error_p = [x for x in error_p if len(x) > 0]
    nerror_p = [x for x in nerror_p if len(x) > 0]
    is_errp = [is_errp[i] for i in range(len(is_errp)) if i not in bad_idx]
    errps = np.array([x[:freq] for x in error_p]).astype(np.float)
    op = np.array([x[:freq] for x in nerror_p]).astype(np.float)
    errps = np.swapaxes(errps, 1, 2)
    op = np.swapaxes(op, 1, 2)
    errps = np.concatenate([errps for _ in range(4)], axis=0)
    # Create Epochs (CTRL)
    action_ps = [[[]] for _ in range(num_actions)]
    ts2_idx = 0
    bad_idx = []
    for i in range(len(fft_data)):
        t = data_ts[i]
        delta = (t - ts2[ts2_idx])
        if delta < 0:
            pass
        elif delta < 2.5 and delta > 0.:
            action_ps[ordered_labels[ts2_idx]][-1].append(fft_data[i])
        elif delta > 2.5:
            if len(action_ps[ordered_labels[ts2_idx]][-1]) < 1.5 * freq:
                action_ps[ordered_labels[ts2_idx]] = action_ps[ordered_labels[ts2_idx]][:-1]
                bad_idx.append(ts2_idx)
            ts2_idx += 1
            for j in range(num_actions):
                action_ps[j].append([])
            if ts2_idx == len(ts2):
                break
    for i in range(num_actions):
        action_ps[i] = [x for x in action_ps[i] if len(x) > 0]
        action_ps[i] =  np.array([x[int(0.5 * freq):int(1.5 * freq)] for x in action_ps[i]]).astype(np.float)
        action_ps[i] = np.swapaxes(action_ps[i], 1, 2)
    ordered_labels = [ordered_labels[i] for i in range(len(ordered_labels)) if i not in bad_idx]
    # Format for network
    X = np.concatenate([op[:int(len(errps) * 1.0)], errps], axis=0)
    labels = [False for x in range(int(len(errps) * 1.0))] + [True for x in errps]
    y = np.array([int(x) for x in labels])
    rnd_ord = np.arange(len(X))
    np.random.shuffle(rnd_ord)
    X = X[rnd_ord]
    y = y[rnd_ord]
    X2 = np.concatenate(action_ps, axis=0)
    labels2 = [[i for _ in range(len(action_ps[i]))] for i in range(num_actions)]
    y2 = np.array([i for sublist in labels2 for i in sublist])
    rnd_ord = np.arange(len(X2))
    np.random.shuffle(rnd_ord)
    X2 = X2[rnd_ord]
    y2 = y2[rnd_ord]
    
    return X, y, X2, y2


def format_datasets(data_idxs=[6, 7, 8], task="erp", channels=8, freq=250):
    datasets = []
    for data_idx in data_idxs:
        X1, y1, X2, y2 = load_data(data_idx)
        if task == "erp":
            X, y = X1, y1
        elif task == "ctrl":
            X, y = X2, y2
        else:
            print("ERROR: unknown task")
            exit()
        X_train = X[:int(0.8 * len(X))] * 1000
        y_train = y[:int(0.8 * len(y))].reshape(-1)
        X_test = X[int(0.8 * len(X)):] * 1000
        y_test = y[int(0.8 * len(y)):].reshape(-1)
        X_train = X_train.reshape(X_train.shape[0], 1, channels, freq) 
        X_test = X_test.reshape(X_test.shape[0], 1, channels, freq)
        y_train = np_utils.to_categorical(y_train)
        y_test  = np_utils.to_categorical(y_test)
        datasets.append({"x_train": X_train, "y_train": y_train, "x_test": X_test, "y_test": y_test})
    return datasets

