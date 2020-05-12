import numpy as np
from tensorflow.keras import utils as np_utils

def load_data(data_idx, freq=125):
    # Load Data
    correct_times = np.load("./data/correct_times_{0}.npy".format(data_idx))
    incorrect_times = np.load("./data/incorrect_times_{0}.npy".format(data_idx))
    left_times = np.load("./data/left_times_{0}.npy".format(data_idx))
    right_times = np.load("./data/right_times_{0}.npy".format(data_idx))
    data = np.load("./data/eeg_data_{0}.npy".format(data_idx))
    fft_data = np.load("./data/eeg_fft_data_{0}.npy".format(data_idx))
    data_ts = np.load("./data/eeg_timestamps_{0}.npy".format(data_idx))
    correct_ts = correct_times.astype(np.float64)
    incorrect_ts = incorrect_times.astype(np.float64)
    left_ts = left_times.astype(np.float64)
    right_ts = right_times.astype(np.float64)
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
    left_idx = 0
    right_idx = 0
    ts2 = []
    is_right = []
    while left_idx < len(left_ts) or right_idx < len(right_ts):
        if left_idx < len(left_ts) and right_idx >= len(right_ts):
            ts2.append(left_ts[left_idx])
            is_right.append(False)
            left_idx += 1
        elif left_idx >= len(left_ts) and right_idx <= len(right_ts):
            ts2.append(right_ts[right_idx])
            is_right.append(True)
            right_idx += 1
        else:
            if left_ts[left_idx] < right_ts[right_idx]:
                ts2.append(left_ts[left_idx])
                is_right.append(False)
                left_idx += 1
            else:
                ts2.append(right_ts[right_idx])
                is_right.append(True)
                right_idx += 1
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
    left_p = [[]]
    right_p = [[]]
    ts2_idx = 0
    bad_idx = []
    for i in range(len(fft_data)):
        t = data_ts[i]
        delta = (t - ts2[ts2_idx])
        if delta < 0:
            pass
        elif delta < 2.5 and delta > 0.:
            if is_right[ts2_idx]:
                right_p[-1].append(fft_data[i])
            else:
                left_p[-1].append(fft_data[i])
        elif delta > 2.5:
            if is_right[ts2_idx]:
                if len(right_p[-1]) < 2.4 * freq:
                    right_p = right_p[:-1]
                    bad_idx.append(ts2_idx)
            else:
                if len(left_p[-1]) < 2.4 * freq:
                    left_p = left_p[:-1]
                    bad_idx.append(ts2_idx)
            ts2_idx += 1
            right_p.append([])
            left_p.append([])
            if ts2_idx == len(ts2):
                break
    right_p = [x for x in right_p if len(x) > 0]
    left_p = [x for x in left_p if len(x) > 0]
    is_right = [is_right[i] for i in range(len(is_right)) if i not in bad_idx]
    right_ps = np.array([x[int(1.4 * freq):int(2.4 * freq)] for x in right_p]).astype(np.float)
    left_ps = np.array([x[int(1.4 * freq):int(2.4 * freq)] for x in left_p]).astype(np.float)
    right_ps = np.swapaxes(right_ps, 1, 2)
    left_ps = np.swapaxes(left_ps, 1, 2)
    # Format for network
    X = np.concatenate([op[:int(len(errps) * 1.0)], errps], axis=0)
    labels = [False for x in range(int(len(errps) * 1.0))] + [True for x in errps]
    y = np.array([int(x) for x in labels])
    rnd_ord = np.arange(len(X))
    np.random.shuffle(rnd_ord)
    X = X[rnd_ord]
    y = y[rnd_ord]
    X2 = np.concatenate([left_ps, right_ps], axis=0)
    labels = [False for x in left_ps] + [True for x in right_ps]
    y2 = np.array([int(x) for x in labels])
    rnd_ord = np.arange(len(X2))
    np.random.shuffle(rnd_ord)
    X2 = X2[rnd_ord]
    y2 = y2[rnd_ord]
    
    return X, y, X2, y2


def format_datasets(data_idxs=[6, 7, 8], task="erp"):
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
        X_train = X_train.reshape(X_train.shape[0], 1, 16, 125) 
        X_test = X_test.reshape(X_test.shape[0], 1, 16, 125)
        y_train = np_utils.to_categorical(y_train)
        y_test  = np_utils.to_categorical(y_test)
        datasets.append({"x_train": X_train, "y_train": y_train, "x_test": X_test, "y_test": y_test})
    return datasets

