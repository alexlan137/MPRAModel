import numpy as np
import pandas as pd

def load(filepath):
    pd.options.mode.chained_assignment = None
    
    bias = 5

    raw_df = pd.read_csv(filepath)
    print(raw_df.info())
    filtered_df = raw_df[['VarID', 'chrom', 'pos', 'ref', 'alt', 'plasmid1_ref', 'plasmid2_ref', 'plasmid3_ref', 'cDNA1_ref', 'cDNA2_ref', 'cDNA3_ref',  'plasmid1_alt', 'plasmid2_alt', 'plasmid3_alt', 'cDNA1_alt', 'cDNA2_alt', 'cDNA3_alt']]
    print(filtered_df.info())
    
    output_df = filtered_df[['VarID', 'chrom', 'pos', 'ref', 'alt']]
    output_df['d1ref'] = (filtered_df['cDNA1_ref'] + bias) / (filtered_df['plasmid1_ref'] + bias)
    output_df['d2ref'] = (filtered_df['cDNA2_ref'] + bias) / (filtered_df['plasmid2_ref'] + bias)
    output_df['d3ref'] = (filtered_df['cDNA3_ref'] + bias) / (filtered_df['plasmid3_ref'] + bias)
    output_df['dref'] = ((output_df['d1ref'] + output_df['d2ref'] + output_df['d3ref'])/3)
    output_df['d1alt'] = (filtered_df['cDNA1_alt'] + bias) / (filtered_df['plasmid1_alt'] + bias)
    output_df['d2alt'] = (filtered_df['cDNA2_alt'] + bias) / (filtered_df['plasmid2_alt'] + bias)
    output_df['d3alt'] = (filtered_df['cDNA3_alt'] + bias) / (filtered_df['plasmid3_alt'] + bias)
    output_df['dalt'] = ((output_df['d1alt'] + output_df['d2alt'] + output_df['d3alt'])/3)
    output_df['delta'] = np.log2(output_df['dalt'] / output_df['dref']) # alt / ref such that a positive delta is a increase in signal 
    print(output_df.head())

    train_df = output_df[['chrom', 'pos', 'ref', 'alt', 'delta']]
    print(train_df.head())

    output_df.to_csv('data/MPRA/processed-abell-filtered.csv')
    train_df.to_csv('data/MPRA/train-abell-filtered.csv')


if ('__main__'):
    load('data/MPRA/GSE174534-Abell-GM12878-filter-to-SNPs.csv')