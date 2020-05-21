import numpy as np
import os
import time
import threading
from src.data_utils import load_data, format_datasets, denoise
from src.meta_learning import reptile
from src.eegnet import EEGNet
from src.bci_env import BCIEnv
from src.sac_bci import SACBCI
from src.bci_trainer import BCITrainer
from src.streamer import Streamer
from src.cursor_ctrl import CursorCtrl
from brainflow.board_shim import BoardShim, BrainFlowInputParams


PROMPT = """What would you like to do?
1) Initialize via meta-learning.
2) Train via imitating an optimal controller.
3) Fine-tune via SAC.
4) Save trained components.
5) Use trained action decoder.
6) Quit.
"""

RETRAIN_META = False

def init_policy():
    parser = BCITrainer.get_argument()
    parser = SACBCI.get_argument(parser)
    parser.set_defaults(test_interval=1000)
    parser.set_defaults(max_steps=4000) # for training on 30 mins of data
    parser.set_defaults(gpu=-1)
    parser.set_defaults(n_warmup=int(0))
    parser.set_defaults(batch_size=64)
    parser.set_defaults(memory_capacity=int(1e4))
    parser.set_defaults(target_update_interval=8000)
    args = parser.parse_args("")
    policy = SACBCI(
            state_shape=(1, 16, 125),
            action_dim=2,
            discount=1.0,
            lr=1e-3,
            batch_size=64,
            memory_capacity=args.memory_capacity,
            n_warmup=args.n_warmup,
            target_update_interval=args.target_update_interval,
            auto_alpha=args.auto_alpha,
            update_interval=4,
            gpu=args.gpu)
    return policy, parser


def main():
    policy, parser = init_policy()
    args = parser.parse_args("")
    reward_dec = EEGNet()
    ctrl_dec = EEGNet()
    value_dec = EEGNet(softmax=False)
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM01MTXZ"
    board = BoardShim(2, params)
    board.prepare_session()
    board.start_stream()
    while True:
        task = input(PROMPT)
        if task.isdigit():
            task = eval(task)
        if task == 1:
            if RETRAIN_META: # assuming balanced datasets
                # ERP
                erp_datasets = format_datasets(task="erp")
                reward_dec.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
                reptile(reward_dec, erp_datasets, save_weights=True, task="erp")
                # CTRL
                ctrl_datasets = format_datasets(task="ctrl")
                ctrl_dec.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
                reptile(ctrl_dec, ctrl_datasets, save_weights=True, task="ctrl")
                # Value
                value_dec.compile(loss='mse', optimizer='adam', metrics = ['mse'])
                for dataset in erp_datasets:
                    dataset["y_train"] = np.abs(dataset["y_train"] - 1)
                    dataset["y_test"] = np.abs(dataset["y_test"] - 1)
                reptile(value_dec, erp_datasets, save_weights=True, task="vf")
            else:
                ctrl_dec = EEGNet()
                value_dec = EEGNet(softmax=False)
                reward_dec.load_weights('./models/erp_meta_init.h5')
                ctrl_dec.load_weights('./models/ctrl_meta_init.h5')
                value_dec.load_weights('./models/vf_meta_init.h5')
                print("Loaded meta-learned weights.")
        elif task == 2:
            data_idx = 20
            print("Collecting data at index {0}.".format(data_idx))
            streamer = Streamer(data_idx=data_idx, board=board)
            cursor_ctrl = CursorCtrl(data_idx=data_idx, streamer=streamer)
            cursor_ctrl.run_game(30 * 60)
            cursor_ctrl.close()
            print("Training ...")
            reward_dec.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
            ctrl_dec.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
            value_dec.compile(loss='mse', optimizer='adam', metrics = ['mse'])
            erp_dataset = format_datasets(data_idxs=[data_idx], task="erp")[0]
            ctrl_dataset = format_datasets(data_idxs=[data_idx], task="ctrl")[0]
            for dp in range(len(erp_dataset["x_test"])):
                erp_dataset["x_test"][dp] = denoise(erp_dataset["x_test"][dp])
            for dp in range(len(erp_dataset["x_train"])):
                erp_dataset["x_train"][dp] = denoise(erp_dataset["x_train"][dp])
            for dp in range(len(ctrl_dataset["x_test"])):
                ctrl_dataset["x_test"][dp] = denoise(ctrl_dataset["x_test"][dp])
            for dp in range(len(ctrl_dataset["x_train"])):
                ctrl_dataset["x_train"][dp] = denoise(ctrl_dataset["x_train"][dp])
            reward_dec.fit(erp_dataset["x_train"],
                           erp_dataset["y_train"],
                           batch_size = 16,
                           epochs = 100, 
                           verbose = 2,
                           validation_data=(erp_dataset["x_test"], erp_dataset["y_test"]))
            ctrl_dec.fit(ctrl_dataset["x_train"],
                         ctrl_dataset["y_train"],
                         batch_size = 16,
                         epochs = 100, 
                         verbose = 2,
                         validation_data=(ctrl_dataset["x_test"], ctrl_dataset["y_test"]))
            erp_dataset["y_train"] = np.abs(erp_dataset["y_train"] - 1)
            erp_dataset["y_test"] = np.abs(erp_dataset["y_test"] - 1)
            value_dec.fit(erp_dataset["x_train"],
                          erp_dataset["y_train"],
                          batch_size = 16,
                          epochs = 100, 
                          verbose = 2,
                          validation_data=(erp_dataset["x_test"], erp_dataset["y_test"]))
            reward_dec.save_weights('./models/erp_d{0}.h5'.format(data_idx))
            ctrl_dec.save_weights('./models/ctrl_d{0}.h5'.format(data_idx))
            value_dec.save_weights('./models/vf_d{0}.h5'.format(data_idx))
            policy.actor.load_weights('./models/ctrl_d{0}.h5'.format(data_idx))
            policy.qf1.load_weights('./models/vf_d{0}.h5'.format(data_idx))
            policy.qf1_target.load_weights('./models/vf_d{0}.h5'.format(data_idx))
            print("Finished training.")
        elif task == 3:
            streamer = Streamer(data_idx=102, board=board)
            cursor_ctrl = CursorCtrl(data_idx=102)
            env = BCIEnv(is_live=True, streamer=streamer, reward_dec=reward_dec, cursor_ctrl=cursor_ctrl)
            for ft_epoch in range(3):
                thread = threading.Thread(target=cursor_ctrl.render_for, args=(300,))
                thread.start()
                print("On Round {0} / 3".format(ft_epoch + 1))
                start_time = time.time()
                epoch_obs = []
                epoch_act = []
                epoch_rew = []
                obs = env.reset()
                while time.time() - start_time < 300.:
                    obs = denoise(obs)
                    epoch_obs.append(obs)
                    action = policy.get_action(obs)
                    time.sleep(2.5)
                    obs, reward, _, _, = env.step(action)
                    epoch_act.append(action)
                    epoch_rew.append(reward)
                epoch_obs.append(obs)
                print("Training ...")
                parser.set_defaults(max_steps=500)
                args = parser.parse_args("")
                rep_buff = {'act': epoch_act, 'obs': epoch_obs, 'rew': epoch_rew}
                trainer = BCITrainer(policy, env, args)
                trainer.rep_buff = rep_buff # pulls data from here instead
                trainer()
            print("Finished fine-tuning.")
            streamer.save_data()
            streamer.close()
            cursor_ctrl.close()
        elif task == 4:
            name = str(int(time.time()))
            os.mkdir('./models/{0}'.format(name))
            reward_dec.save_weights('./models/{0}/reward_dec.h5'.format(name))
            policy.actor.save_weights('./models/{0}/policy_actor.h5'.format(name))
            policy.qf1.save_weights('./models/{0}/policy_qf1.h5'.format(name))
            policy.qf1_target.save_weights('./models/{0}/policy_qf1_target.h5'.format(name))
            policy.qf2.save_weights('./models/{0}/policy_qf2.h5'.format(name))
            policy.qf2_target.save_weights('./models/{0}/policy_qf2_target.h5'.format(name))
            print('Saved weights with prefix {0}'.format(name))
        elif task == 5:
            duration = 60
            print("Starting up CursorCtrl for 1 minute.")
            streamer = Streamer(data_idx=103, board=board)
            cursor_ctrl = CursorCtrl(data_idx=103)
            env = BCIEnv(is_live=True, streamer=streamer, reward_dec=reward_dec, cursor_ctrl=cursor_ctrl)
            thread = threading.Thread(target=cursor_ctrl.render_for, args=(duration,))
            thread.start()
            start_time = time.time()
            all_obs = []
            obs = env.reset()
            while time.time() - start_time < duration:
                obs = denoise(obs)
                all_obs.append(obs)
                action = policy.get_action(obs, test=True)
                time.sleep(2.5)
                obs, _, _, _,= env.step(action)
            np.save("all_obs_20.npy", all_obs)
        elif task == 6:
            exit()
        else:
            print("Sorry, command not understood.")



if __name__ == '__main__':
    main()