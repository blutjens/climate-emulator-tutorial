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
You can read the abstract as pdf [here](docs/agu23_lutjens_tutorial_on_fast_climate_emulators.pdf).
```
@misc{lutjens23climatetutorial,
    author = {Lütjens, Björn and Hadzic, Lea M. and Newman, Dava and Veillette, Mark},
    publisher = {AGU23 Fall Meeting},
    title={The Climate Pocket: Tutorial on Building Fast Emulators in Climate Modeling},
    year={2023},
    url={https://agu.confex.com/agu/fm23/meetingapp.cgi/Paper/1304372}
}
```
