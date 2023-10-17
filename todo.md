# List of todo's

[] test environment.yml
model development
    [x] download data locally to avoid google colab...
    [x] declutter notebook by moving unet_model into python files
    [x] declutter climate-emulator-tutorial notebook by moving normalize to python files
    [x] declutter tutorial by moving autoregressive dataset into python files
    [x] fit baseline global co2 -> global T -> local T using linear regression.
    [] write train.py s.t., i can plug in an fcnn, unet, etc. given a dataloader?
        [] rewrite dataloader, s.t., the explore_data scipt saves an ML-ready dataset in processed/. The train.py then takes in a path to that dataset and creates a dataloader.
        [] rewrite criterion s.t., it's an argument in cfg
        [] maybe get rid of click.
    [] im optimizing too much at the same time. What do I want to do first?
        [x] map global cum. co2, ch4, bc, and so2 at t to global T at t
            reshape into list of data samples or large numpy array?
                numpy array easy for linear regression.
                list easy for fcnn data loader
                -> do a list and write linear regression to pull all data first.
                WHY IS THIS SO COMPLICATED?!
                    i have multiple scenarios who's information i dont want to lose
                    i have multiple timestamps who's information i dont want to lose.
                    **i need to find a library for processing video data?**
    [] write extensible code for ghg -> cmip output mapping
        write it as dataset_utils: interim_to_processed.py
        inputs - batch, c, h, w
            e.g., global cum. co2, ch4, bc, so2 at t
            e.g., global cum. co2, ch4 and local bc, so2 at t
            (e.g., global cum. co2, ch4 over t-10:t and local bc, so2)
            e.g., global cum. co2, ch4, local bc, so2 at t and 
                local T, huss, ps at t-1
            e.g., global cum. co2, ch4, local bc, so2 at t and
                local T, huss, ps at t-3:t-1
        outputs - batch, c, h, w
            e.g., global T at t
            e.g., local T at t
            e.g., local T, huss, ps at t
            e.g., local T, huss , ps at t+3
    [] can ML do at least as good as current emulators?    
        [] *Linear: map annual cum. CO2 emission + (CH4,SO2,BC) emissions at time t linearly to global T at t. then map global T to local T*
        [] FaiRGP: map annual cum. CO2 emissions + (CH4,SO2,BC) emissions at time t to global T; then map global T to local T
    [] ClimateBench: map global cum. CO2 emissions + global CH4 emis. + spatially-resolved annual (SO2,BC) at time t to local T
    [] Nat-Var-FCNN: map 10yr cum. co2 + (CH4, SO2, BC) time-series to 10 yr avg local T) 

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
            * it seems like all errors - even in arctic - are just the result of internal variability?! * -> maybe, maybe not.
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

