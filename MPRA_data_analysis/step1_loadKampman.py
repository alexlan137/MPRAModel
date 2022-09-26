import pandas as pd

def loadKampman(filepath, pval, p_threshold, id, loaded_dir):
    pd.options.mode.chained_assignment = None
    raw_df = pd.read_csv(filepath)
    train_df = raw_df[['vars', 'seq', 'seq2', 'plasmid1_ref', 'plasmid2_ref', 'plasmid3_ref', 'plasmid1_alt', 'plasmid2_alt', 'plasmid3_alt', 'log2FoldChange_allele', 'padj_allele']] #seq length = 300bp
    train_df = train_df[train_df['plasmid1_ref'] + train_df['plasmid2_ref'] + train_df['plasmid3_ref'] > p_threshold]
    train_df = train_df[train_df['plasmid1_alt'] + train_df['plasmid2_alt'] + train_df['plasmid3_alt'] > p_threshold]
    train_df = train_df[train_df['padj_allele'] < pval]
    train_df['delta'] = train_df['log2FoldChange_allele']
    print(train_df.info())
    print(train_df.head())
    train_df.to_csv(loaded_dir + id + '.csv')