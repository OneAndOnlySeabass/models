# This script generates a stratified random subset of n size from a Kepler TCE csv
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--"--input_tce_csv_file",
    type=str,
    required=True,
    help="CSV file containing the Q1-Q17 DR24 Kepler TCE table. Must contain "
    "columns: rowid, kepid, tce_plnt_num, tce_period, tce_duration, tce_time0bk.") 

parser.add_argument(
    "--sub_size",
    type=int,
    required=True,
    help="Size of the desired subset"
    
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory in which to save the output.")