import argparse
import time
import numpy as np
import pandas as pd

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import mne

# https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#streaming-board

def main ():
    BoardShim.enable_dev_board_logger ()

    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM01MTXZ"
    board_id = 2
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(1830) # can do this in a loop to deal w/ live stuff hmmmm
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    ts_channels = BoardShim.get_timestamp_channel(board_id)
    eeg_data = data[eeg_channels, :]
    eeg_ts = data[ts_channels, :]
    eeg_data = eeg_data / 1000000
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    ch_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    sfreq = BoardShim.get_sampling_rate (board_id)
    info = mne.create_info (ch_names = ch_names, sfreq = sfreq, ch_types = ch_types)
    raw = mne.io.RawArray (eeg_data, info)
    raw2 = mne.io.RawArray (np.copy(eeg_data), info)
    raw.filter(0.5, 40., fir_design='firwin')
    raw2.filter(1.0, 60., fir_design='firwin')
    np.save("eeg_data_8.npy", np.transpose(raw.get_data()))
    np.save("eeg_fft_data_8.npy", np.transpose(raw2.get_data()))
    np.save("eeg_timestamps_8.npy", eeg_ts)


if __name__ == "__main__":
    main ()