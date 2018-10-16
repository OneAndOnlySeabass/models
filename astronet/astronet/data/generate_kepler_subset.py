# Written by Sebastiaan Koning (GitHub: OneAndOnlySeabass) 15-10-2018
# This script generates a stratified random subset of n size from a Kepler TCE csv.

import pandas as pd
import numpy as np

# Adjustable variables can be changed here
read_loc = #r"tce csv location"
pc_subset = 1000
fp_subset = 1000 # Both AFPs and NTPs
write_loc = #r"desired output csv location"

# Reading the csv file from read_loc
kepler_df = pd.read_csv(read_loc, index_col="rowid", comment="#")
                                       
# Removing rows with av_training_set=='UNK'
kepler_df = kepler_df[kepler_df.av_training_set != 'UNK']

# Dividing the dataset in PCs and FPs(AFPs & NTPs)
PC_df = kepler_df[kepler_df.av_training_set == 'PC']
FP_df = kepler_df[kepler_df.av_training_set != 'PC']

# Random selection of 1000 PCs and 1000 NPs
np.random.seed(114639)
PC_random = PC_df.sample(n=pc_subset)
FP_random = FP_df.sample(n=fp_subset)

sample_df = pd.concat((PC_random, FP_random))
sample_df = sample_df.sample(frac=1) # Shuffles the data

# Writing a new csv to write_loc
sample_df.to_csv(write_loc, index=False)