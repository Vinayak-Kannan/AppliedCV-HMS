This is the repo of our COMS 4995 Applied Computer Vision semester project: an ensemble method classification tool for EEG diagnoses.

Authors: Raman Odgers, Akhil Golla, Vinayak Kannan, Sohan Kshirsagar

# Github: AppliedCV-HMS
This repo includes all relevant exploratory, data processing, and training files.
The files here are not intended to be run but to show the work by the team to build our final product model.

`fusion[config,training,utils].py` and `modeltest.py` Are the final model train and test files. 

Fusiontraining is a simple wrapper file that instantiates relevant input models, and trains by calling methods from Fusionutils. Fusionconfig sets relevant paths and hyperparameters for the training process. 

Modeltest contains our testing pipeline. 

Rocket and xgboost directories contain respective model files. 

The EfficientNet and Ensemble model files are hosted in a Model repo on HuggingFace [here](https://huggingface.co/ramanodgers/HMSensemble).

`FMdiagram.png` displays the computation graph of the final ensemble model. 

Training can be run simply with `python fusiontraining.py`

# Huggingface: HMSDemo
We also have a live deployment for our model, hosted on a Huggingface Space [here](https://huggingface.co/spaces/ramanodgers/HMSDemo). 
This is synced with another github repo for a smoother workflow: [here](https://github.com/ramanodgers/HMSDemo).

You should be able to drag and drop an EEG and spectrogram file on the huggingface space and receive an accurate EEG classification. 
We provide an 'example_eeg.parquet' file for this purpose in the demo repository.  

