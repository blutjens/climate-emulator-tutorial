# climate-emulator-tutorial
A tutorial for getting started with machine learning for climate modeling

Edit notebook at: https://colab.research.google.com/drive/1XwTghkLbxrckwUeE_UMQ-HtuAeYfEETj?usp=sharing

### Run hyperparameter search on server
Note: this script has only been tested on a supercomputer at MIT. Modifications are likely necessary for other supercomputers.
```
# gpu on supercloud tested and works.
ssh <user>@txe1-login.mit.edu
conda deactivate
module load anaconda/2023a-pytorch #  on engaging eofe7: module load anaconda3/2020.11
tmux new -s emcli # Can be reactivated with: tmux at -t emcli
# Activate tmux mouse scrolling via ^b + : then "setw -g mouse on" # or mode-mouse
LLsub -i -s 4 -T 02:00:00 -g volta:1 # LLsub -i full #  On engaging eofe7 run $$srun -p sched_mit_darwin2 -n 28 --mem-per-cpu=4000 --pty /bin/bash # -N 1
conda activate emcli
# might need: source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a-pytorch/etc/profile.d/conda.sh
export WANDB_MODE='offline'
python emcli/models/unet/train.py --parallel
--> this will import for 10min
wandb sync latest-run
```
