# Supplementory code: 
This is supplementory code repository for paper Impedance-based equivalent circuit model structure and parameter update for electric vehicles using machine learning.
This repository include how to reproduce result. 
## Quick start
### (1) Set up python envs
```
pip install -r requirements.txt
```
### (2) Download raw dataset (refer to 10. Data availability in paper)
#### The updated raw data is in text file, please first transfer the .txt file into excel file.
### (3) Preprocess dataset
```
python preprocess.py --sheet1 $FULL_PATH_TO_A --sheet2 $FULL_PATH_TO_B
```
#### $FULL_PATH_TO_A refer to full path to your data for model a, $FULL_PATH_TO_B refer to full path to your data for model b.

### (4) Regression reproduction
```
python regression.py --model b
```
#### Please change model to b if you want to get result for model b

### (5) Classification reproduction
```
python classification.py
```

## Randomness
Please note that the dataset split is randomly generated (as in many standard ML/DL workflows), which may lead to run-to-run variation in the reported results. To ensure fair comparison, we repeat each experiment multiple times and report the median performance.
