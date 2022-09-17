import pandas as pd

def load(filepath):
    pd.options.mode.chained_assignment = None
    raw_df = pd.read_csv(filepath)
    filtered_df = raw_df[['VarID', 'chrom', 'pos', 'ref', 'alt', 'log2FoldChange_allele']]
    train_df = filtered_df[['chrom', 'pos', 'ref', 'alt']]
    train_df['delta'] = filtered_df['log2FoldChange_allele']
    print(train_df.info())
    print(train_df.head())
    train_df.to_csv('data/MPRA_loaded/AbellV2.csv')

if ('__main__'):
    load('data/MPRA/AbellV2.csv')