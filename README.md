
# Goad-driven Trajectory prediction

This is the supplementary code for the WACV submission 804. It contains the core source code for reviewers to get the reported results.

## Setup

The code was developed and tested on Ubuntu 18.04 LTS using Python 3.8.

Before running the code, you need to install Anaconda to create python virutal environment.

Assuming that Anaconda has been installed, run the following command install the required dependencies.

```
conda env create -f environment.yml

conda activate gtp
```

## Download data and pretrained models

Download the [data](https://drive.google.com/file/d/1swIk1Mn8asEzJ1ZkNH5K4GAJWPT5kmAF/view?usp=sharing) and [pretrained model](https://drive.google.com/file/d/1ds88deyzqQwSb_MrfVw1JEzHtpon2Nht/view?usp=sharing).

Put the two download files to the two downloaded files to the downloads/ folder, then run the following command

```
chmod +x scripts/extract_downloaded_files.sh

bash scripts/extract_downloaded_files.sh
```

## Run models
```
chmod +x scripts/get_all_results.sh

bash scripts/get_all_results.sh
```