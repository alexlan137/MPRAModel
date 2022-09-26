import pandas as pd

def loadAbell(filepath, pval, p_threshold, c_threshold, id, loaded_dir):
    pd.options.mode.chained_assignment = None
    raw_df = pd.read_csv(filepath)
    filtered_df = raw_df[['VarID', 'chrom', 'pos', 'ref', 'alt', 'plasmid1_ref', 'plasmid2_ref', 'plasmid3_ref', 'plasmid1_alt', 'plasmid2_alt', 'plasmid3_alt', 'cDNA1_ref', 'cDNA2_ref', 'cDNA3_ref', 'cDNA1_alt', 'cDNA2_alt', 'cDNA3_alt','log2FoldChange_allele', 'padj_allele']]
    
    # plasmid1_ref = filtered_df['plasmid1_ref']
    # plasmid2_ref = filtered_df['plasmid2_ref']
    # plasmid3_ref = filtered_df['plasmid3_ref']
    # plasmid1_alt = filtered_df['plasmid1_alt']
    # plasmid2_alt = filtered_df['plasmid2_alt']
    # plasmid3_alt = filtered_df['plasmid3_alt']
    
    filtered_df = filtered_df[filtered_df['plasmid1_ref'] + filtered_df['plasmid2_ref'] + filtered_df['plasmid3_ref'] > p_threshold]
    filtered_df = filtered_df[filtered_df['plasmid1_alt'] + filtered_df['plasmid2_alt'] + filtered_df['plasmid3_alt'] > p_threshold]
    filtered_df = filtered_df[filtered_df['cDNA1_ref'] + filtered_df['cDNA2_ref'] + filtered_df['cDNA3_ref'] > c_threshold]
    filtered_df = filtered_df[filtered_df['cDNA1_alt'] + filtered_df['cDNA2_alt'] + filtered_df['cDNA3_alt'] > c_threshold]
    filtered_df = filtered_df[filtered_df['padj_allele'] < pval]
    train_df = filtered_df[['chrom', 'pos', 'ref', 'alt', 'plasmid1_ref', 'plasmid2_ref', 'plasmid3_ref', 'plasmid1_alt', 'plasmid2_alt', 'plasmid3_alt']]
    train_df['delta'] = filtered_df['log2FoldChange_allele']
    print(train_df.info())
    print(train_df.head())
    train_df.to_csv(loaded_dir + id + '.csv')
    
    # plasmid1_ref.to_csv('data/MPRA_loaded/AbellV2.plasmid1_ref')
    # plasmid2_ref.to_csv('data/MPRA_loaded/AbellV2.plasmid2_ref')
    # plasmid3_ref.to_csv('data/MPRA_loaded/AbellV2.plasmid3_ref')
    # plasmid1_alt.to_csv('data/MPRA_loaded/AbellV2.plasmid1_alt')
    # plasmid2_alt.to_csv('data/MPRA_loaded/AbellV2.plasmid2_alt')
    # plasmid3_alt.to_csv('data/MPRA_loaded/AbellV2.plasmid3_alt')
