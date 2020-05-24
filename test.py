import dataloader
import visualizer
import model
from recon import evaluate
from utility import getAbs, getPhase

device_no = 1
torch.cuda.set_device(device_no)
device = torch.device("cuda:"+str(device_no) if torch.cuda.is_available() else "cpu")

experimental_data_path = '../../DATA/exp_mc_amp_usaf_2018_11_23/USAF_amplitude_dataset.mat'
training_data_path = '/home/kellman/Workspace/PYTHON/Design_FPM_pytorch/datasets_train_iccp_results/train_amp_exp_n10000.mat'
ckpt_path = './runs/08:49:07_batch_size=5_stepsize=0.005_loss_fn=abs_optim=adam_num_unrolls=100_alpha=0.100_num_df=4_num_bf=1_num_leds=89/ckpt.tar'
