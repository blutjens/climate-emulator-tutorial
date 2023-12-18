# climate-emulator-tutorial
A tutorial for getting started with machine learning for climate modeling. To read the tutorial go to:

[climate_emulator_tutorial.ipynb](https://nbviewer.org/github/blutjens/climate-emulator-tutorial/blob/main/climate_emulator_tutorial.ipynb)

# Installation
```
git clone git@github.com:blutjens/climate-emulator-tutorial.git
cd climate-emulator-tutorial
conda create --name emcli
conda activate emcli
conda install pip
pip install -r requirements.txt
pip install -e .
ipython kernel install --user --name=emcli # Link conda environment to jupyter notebook
```

# Start the notebook
```
jupyter notebook climate_emulator_tutorial.ipynb
```

# Explore the demo
```
# Explore demo online at
https://climate-emulator-tutorial.streamlit.app/
# Develop demo locally with
conda activate emcli
streamlit run run_climate_pocket_webapp.py
```

# Reference
```
@misc{lutjens23climatetutorial,
    author = {Lütjens, Björn and Hadzic, Lea M. and Newman, Dava and Veillette, Mark},
    publisher = {AGU23 Fall Meeting},
    title={The Climate Pocket: Tutorial on Building Fast Emulators in Climate Modeling},
    year={2023},
    url={https://agu.confex.com/agu/fm23/meetingapp.cgi/Paper/1304372}
}
```

## Miscellanous likely irrelevant:
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

### (Untested scripts:) Running the code on MIT's svante supercomputer
```
ssh <kerberos>@svante-login.mit.edu # or to land on file server $ ssh <kerberos>@svante6.mit.edu
cd /net/fs06/d3/CMIP5/MPI-GE/
module load anaconda3/2023.07 # request pytorch as software by emailing jp-admin@techsquare.com
srun --pty -p fdr -n 1 /bin/bash # interactive node
source activate emcli
# has packages: conda install jupyterlab, cartopy
# need packages: conda install netcdf4, torchvision, (jupyter_bokeh, geoviews), regionmask, streamlit, xskillscore
conda deactivate
```
