import numpy as np 
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import mne


class Streamer:
    def __init__(self, data_idx=101):
        self.data_idx = data_idx
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-DM01MTXZ"
        self.board_id = 2
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        sfreq = BoardShim.get_sampling_rate(self.board_id)
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
        ch_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        self.info = mne.create_info (ch_names = ch_names, sfreq = sfreq, ch_types = ch_types)
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()
    def get_data(self, num_samples, freq_ranges):
        data = self.board.get_board_data()
        eeg_data = data[self.eeg_channels, :] 
        eeg_data = eeg_data / 1000000
        samples = []
        for (low, high) in freq_ranges:
            raw = mne.io.RawArray(eeg_data, self.info)
            raw.filter(low, high, fir_design='firwin')
            samples.append(np.swapaxes(np.transpose(raw.get_data()), 1, 2))
        return samples
    def save_data(self):
        data = self.board.get_board_data()
        ts_channels = BoardShim.get_timestamp_channel(self.board_id)
        eeg_data = data[self.eeg_channels, :]
        eeg_ts = data[ts_channels, :]
        eeg_data = eeg_data / 1000000
        raw = mne.io.RawArray (eeg_data, self.info)
        raw2 = mne.io.RawArray (np.copy(eeg_data), self.info)
        raw.filter(0.5, 40., fir_design='firwin')
        raw2.filter(1.0, 60., fir_design='firwin')
        np.save("./data/eeg_data_{0}.npy".format(self.data_idx), np.transpose(raw.get_data()))
        np.save("./data/eeg_fft_data_{0}.npy".format(self.data_idx), np.transpose(raw2.get_data()))
        np.save("./data/eeg_timestamps_{0}.npy".format(self.data_idx), eeg_ts)
    def close(self):
        self.board.stop_stream()
        self.board.release_session()