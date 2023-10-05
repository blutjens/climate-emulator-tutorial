# List of todo's

[] test environment.yml
[x] download data locally to avoid google colab...
[x] declutter notebook by moving autoregressiveDataset and unet_model into python files
[] data exploration:
    [] How to deal with internal variability?
        [] could smoothen global T and then map co2 onto global T
    [] When does the linear pattern scaling break down?
        [] In which regions does the linear pattern scaling break down?
            [x] create 2D map plot of (pattern scaling - ground-truth local tas) on the avg in 2080-2100.
	        [x] plot error for 26 SREX regions (only land) or 58 AR6-WGI regions, e.g., Arctic, Monsoon region,  
                -> see tutorial of AR6-WGI at: https://github.com/SantanderMetGroup/ATLAS/blob/main/notebooks/reference-regions_Python.ipynb
            [x] plot regional grid according to notebook, [7], with label "abbrev", add_ocean=True
            [x] plot global T and local T over time in Arctic (then in Monsoon, etc.) -> done!
            * it seems like all errors - even in arctic - are just the result of internal variability?! *
        [] Does the linear assumption break down during rare events?
        could plot accuracy (y-axis) over pixel-wise binned TAS prediction value, where TAS prediction values are ordered according to frequency (or TAS if TAS itself is Gaussian distributed)
    [] get so2 and bc -- where are they?
	[] plot ch4, so2, and bc over time like climatebench fig1 
    [x] plot pixel wise frequency of TAS values
        - calculate running sum of histogram over all images TAS images
    [x] plot tas histogram in test set
    [x] plot all 740 training data samples. I want to know if I can learn as a human how to do the prediction task. 
        [] find out why 'tas' runs from -3 to +5? Am I plotting delta tas? But at which point have I dont the conversion.
        [] plot spatially-resolved 20year average tas on right and global co2 concentrations over time on left. The question is if I can see a mapping in between the co2 and local tas values?
    [] create scatterplot of co2 to each pixel? to see if co2 relates locally linearly to tas
    [] create 2nd order polynomial fit instead of pattern scaling. I'm interested if there's good trends in the global T vs. local T plot.

model development
    [] write train.py s.t., i can plug in an fcnn, unet, etc. given a dataloader?
        [] rewrite dataloader, s.t., the explore_data scipt saves an ML-ready dataset in processed/. The train.py then takes in a path to that dataset and creates a dataloader.
        [] rewrite criterion s.t., it's an argument in cfg
        [] maybe get rid of click.
	[] FaiRGP: map annual cum. CO2 emissions + (CH4,SO2,BC) emissions at time t to global T; then map global T local T
	[] ClimateBench: map global cum. CO2 emissions + global CH4 emis. + spatially-resolved annual (SO2,BC) at time t to local T
	[] Linear: map annual cum. CO2 emission + (CH4,SO2,BC) emissions at time t to global T.
		then map global T to local T
	[] Nat-Var-FCNN: map 10yr cum. co2 + (CH4, SO2, BC) time-series to 10 yr avg local T) 