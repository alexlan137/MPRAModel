import pandas as pd

def load(filepath):
    pd.options.mode.chained_assignment = None
    raw_df = pd.read_csv(filepath)
    filtered_df = raw_df[['vars', 'seq', 'seq2', 'log2FoldChange_allele', 'padj_allele']] #seq length = 300bp
    # filtered_df = filtered_df[filtered_df['padj_allele'] < 0.1]
    train_df = filtered_df[['seq', 'seq2']]
    train_df['delta'] = filtered_df['log2FoldChange_allele']
    print(train_df.info())
    print(train_df.head())
    filtered_df.to_csv('data/MPRA_loaded/Kampman.fullinfo.csv')
    train_df.to_csv('data/MPRA_loaded/Kampman.csv')

if ('__main__'):
    load('data/MPRA/KampmanK562.csv')