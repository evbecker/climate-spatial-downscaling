# Downscaling Project

### Requirements and Setup
Necessary python packages can be found in environment.yaml and installed with conda directly. 
Note: *xarray* and *cartopy* are only used for preprocessing and visualization, and can be skipped if only doing training and evaluation. 

### Dataset
Netcdf and torch tensor files currently located in private directory [here](https://drive.google.com/drive/folders/1Z58DqsE5lK8XeUaEb1_TO6YAagzXqaws?usp=sharing). They consist of ERA-iterim, NOAA CPC, and high resolution WRF datasets. 
Please email evbecker @ ucla.edu for access or download manually from:
* [WRF](https://rda.ucar.edu/datasets/ds612.0)
* [ERA-Iterim](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-interim)
* [CPC](https://psl.noaa.gov/data/gridded/data.unified.daily.conus.html), [CPC (Real Time)](https://psl.noaa.gov/data/gridded/data.unified.daily.conus.rt.html)

### Directory Overview
*  **data_utils:** Contains scripts for preprocessing and visualizing data.
	* preprocessing.py: converts raw netcdf data to daily image tensors
*  **models:** Defines the base modules used in the conditional GAN
* **skdownscale:** Contains code for traditional downscaling baselines
* embedding_analysis.py: plots principal components of image embeddings
* visualize_predictions.py: plots model predictions along with low and high resolution data
* dataloader.py: defines pytorch datasets to be used during training
* gan_train.py: runs the script to train the GAN models
* generate_results.py: generates the results from the trained networks and save them as .pt files in the designated folder
* numerical_analysis.py: perform quantative analysis of the given model

### Training
To train the GAN model, simply run gan_train.py. Note that in the script, dataset can be set to 40 or 160, where 40 indicates using ERA as low resolution data and CPC as high resolution data while 160 indicates using CPC as low resolution data and WRF as high resolution data. mode can be set to 'ours' or 'EAD', where 'ours' is to train our model and 'EAD' is to train the EAD model. l1_lambda can be set to any value, which represent the weight for the reconstruction loss.

### Evaluation
For the quantitative results, use the evaluation() function in numerical_analysis.py. The first argument indicates what method to evaluate, which can be 'ours', 'EAD', 'AE', and 'naive'. The second argument points to the location where the model parameters are stored, which should be a .pt file. The third argument decides if the network was trained on the 40x40 data (ERA->CPC) or the 160x160 data (CPC->WRF) (40 for 40x40 data and 160 for 160x160 data). 

To generate the results, use the generate_results.py, where the reso can be set to 40 or 160 which indicates what data the network was trained on. The mode in the script defines what method to use, which can be 'ours', 'EAD', 'AE', and 'naive'. The res_folder variable is the path to the folder you want to save your results in. Also, please remember to change the model path if different models are used.
