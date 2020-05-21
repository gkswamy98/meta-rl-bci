import numpy as np 
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import mne


class Streamer:
    def __init__(self, data_idx=101, board=None):
        self.data_idx = data_idx
        self.board_id = 2
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.ts_channels = BoardShim.get_timestamp_channel(self.board_id)
        sfreq = BoardShim.get_sampling_rate(self.board_id)
        self.freq = sfreq
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
        ch_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        self.info = mne.create_info (ch_names = ch_names, sfreq = sfreq, ch_types = ch_types)
        if board is not None:
            self.board = board
        else:
            params = BrainFlowInputParams()
            params.serial_port = "/dev/cu.usbserial-DM01MTXZ"
            board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            self.board.start_stream()
    def get_data(self, num_samples, freq_ranges, time=False):
        data = self.board.get_current_board_data(60 * self.freq)
        eeg_data = data[self.eeg_channels, :] 
        eeg_data = eeg_data / 1000000
        samples = []
        for (low, high) in freq_ranges:
            raw = mne.io.RawArray(np.copy(eeg_data), self.info)
            raw.filter(low, high, fir_design='firwin')
            sample = raw.get_data()[:, -num_samples:]
            samples.append(np.expand_dims(sample, 0))
        if time:
            return samples, data[self.ts_channels, -num_samples:]
        else:
            return samples
    def save_data(self):
        data = self.board.get_current_board_data(self.board.get_board_data_count())
        ts_channels = BoardShim.get_timestamp_channel(self.board_id)
        eeg_data = data[self.eeg_channels, :]
        eeg_ts = data[ts_channels, :]
        eeg_data = eeg_data / 1000000
        np.save("raw_data.npy", eeg_data)
        raw = mne.io.RawArray (eeg_data, self.info)
        raw2 = mne.io.RawArray (np.copy(eeg_data), self.info) # do this if need be ...
        raw.filter(0.5, 40., fir_design='firwin')
        raw2.filter(9., 50., fir_design='firwin')
        data1 = raw.get_data()
        data2 = raw2.get_data()
        np.save("./data/eeg_data_{0}_old.npy".format(self.data_idx), np.transpose(data1))
        np.save("./data/eeg_fft_data_{0}_old.npy".format(self.data_idx), np.transpose(data2))
        np.save("./data/eeg_timestamps_{0}_old.npy".format(self.data_idx), eeg_ts)
    def close(self):
        self.board.stop_stream()
        self.board.release_session()