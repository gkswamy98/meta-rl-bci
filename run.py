import numpy as np
from src.data_utils import load_data, format_datasets
from src.meta_learning import reptile
from src.eegnet import EEGNet
from src.bci_env import BCIEnv
from src.sac_bci import SACBCI
from src.bci_trainer import BCITrainer
from src.streamer import Streamer

PROMPT = """What would you like to do?
1) Initialize via meta-learning.
2) Train via imitating an optimal controller.
3) Fine-tune via SAC.
4) Use trained action decoder.
5) Quit.
"""

RETRAIN_META = False

def init_policy():
    parser = BCITrainer.get_argument()
    parser = SACBCI.get_argument(parser)
    parser.set_defaults(test_interval=1000)
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(gpu=-1)
    parser.set_defaults(n_warmup=int(2e4))
    parser.set_defaults(batch_size=64)
    parser.set_defaults(memory_capacity=int(1e4))
    parser.set_defaults(target_update_interval=8000)
    args = parser.parse_args("")
    policy = SACBCI(
            state_shape=(1, 16, 125),
            action_dim=2,
            discount=1.0,
            lr=3e-4,
            batch_size=64,
            memory_capacity=args.memory_capacity,
            n_warmup=args.n_warmup,
            target_update_interval=args.target_update_interval,
            auto_alpha=args.auto_alpha,
            update_interval=4,
            gpu=args.gpu)
    return policy


def main():
    policy = init_policy()
    reward_dec = EEGNet()
    env = None
    while True:
        task = input(PROMPT)
        if task.isdigit():
            task = eval(task)
        if task == 1:
            if RETRAIN_META: # assuming balanced datasets
                # ERP
                erp_datasets = format_datasets(task="erp")
                reward_dec.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
                reptile(reward_dec, ctrl_datasets, save_weights=True, task="erp")
                # CTRL
                ctrl_dec = EEGNet()
                ctrl_datasets = format_datasets(task="ctrl")
                ctrl_dec.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
                reptile(ctrl_dec, ctrl_datasets, save_weights=True, task="ctrl")
                # Value
                value_dec = EEGNet(softmax=False)
                value_dec.compile(loss='mse', optimizer='adam', metrics = ['mse'])
                for dataset in erp_datasets:
                    dataset["y_train"] = np.abs(dataset["y_train"] - 1)
                    dataset["y_test"] = np.abs(dataset["y_test"] - 1)
                reptile(value_dec, erp_datasets, save_weights=True, task="vf")
            else:
                reward_dec.load_weights('./models/erp_meta_init.h5')
                policy.actor.load_weights('./models/ctrl_meta_init.h5')
                policy.qf1.load_weights('./models/vf_meta_init.h5')
                policy.qf1_target.load_weights('./models/vf_meta_init.h5')
                print("Loaded meta-learned weights.")
        elif task == 2:
            streamer = Streamer(101)
            env = None # Instead of BCI Env, create GameEnv (no need for policy)
            # Afterwards, generate not-live bci_env and train for some number of timesteps
            # the idea of a live bci env was just for conveniece, should remove.
            # TODO
            streamer.save_data() # for future meta-learning
            streamer.close()
        elif task == 3:
            pass
        elif task == 4:
            pass
        elif task == 5:
            exit()
        else:
            print("Sorry, command not understood.")



if __name__ == '__main__':
    main()