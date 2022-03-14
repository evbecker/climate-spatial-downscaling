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
* gan_train.py:
* generate_results.py:
* numerical_analysis.py:

### Training

### Evaluation