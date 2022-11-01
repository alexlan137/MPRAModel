import pandas as pd
import pysam

def insert_variant(seq, allele, position):
    """ Inserts the specified allele at the given position of the sequence
    """
    left, right = seq[:position-1], seq[position:]
    return left + allele + right

def loadAbellK(filepath, pval, p_threshold, id, loaded_dir):
    pd.options.mode.chained_assignment = None
    raw_df = pd.read_csv(filepath)
    train_df = raw_df[['VarID', 'chrom', 'pos', 'ref', 'alt', 'plasmid1_ref', 'plasmid2_ref', 'plasmid3_ref', 'plasmid1_alt', 'plasmid2_alt', 'plasmid3_alt', 'log2FoldChange_allele', 'padj_allele']] #seq length = 150bp
    train_df = train_df[train_df['plasmid1_ref'] + train_df['plasmid2_ref'] + train_df['plasmid3_ref'] > p_threshold]
    train_df = train_df[train_df['plasmid1_alt'] + train_df['plasmid2_alt'] + train_df['plasmid3_alt'] > p_threshold]
    train_df = train_df[train_df['padj_allele'] < pval]
    train_df['delta'] = train_df['log2FoldChange_allele']
    seqR = []
    seqA = []
    fasta_ref = pysam.FastaFile('reference/hg38.genome.fa')

    for idx, row in train_df.iterrows():
        
        chrom = str(row['chrom'])
        peak_loc = 75
        start = int(row['pos'] - peak_loc) # index? should be right based on SNP matching
        end = int(row['pos'] + peak_loc)
        seq = fasta_ref.fetch(chrom, start, end).upper()
        ref = row['ref']
        alt = row['alt']
        if (idx % 10000 == 0):
            print(ref, alt, seq[75])
        bases = ['A', 'C', 'G', 'T']
        if (ref not in bases or alt not in bases):
            print("ERROR")
            print(ref, alt, seq[75])
            continue
        refseq = insert_variant(seq, ref, peak_loc)
        altseq = insert_variant(seq, alt, peak_loc)
        if(len(seq)!= 150):
            seqR.append("ERROR")
            seqA.append("ERROR")
            continue
        seqR.append(refseq)
        seqA.append(altseq)
    train_df['seq'] = seqR
    train_df['seq2'] = seqA
    train_df = train_df[train_df['seq'] != "ERROR"]
    print(train_df.info())
    print(train_df.head())
    train_df.to_csv(loaded_dir + id + '.csv')

def loadAbellKScaled(filepath, pval, p_threshold, id, loaded_dir):
    pd.options.mode.chained_assignment = None
    raw_df = pd.read_csv(filepath)
    train_df = raw_df[['VarID', 'chrom', 'pos', 'ref', 'alt', 'plasmid1_ref', 'plasmid2_ref', 'plasmid3_ref', 'plasmid1_alt', 'plasmid2_alt', 'plasmid3_alt', 'lfc_scaled', 'padj_allele']] #seq length = 150bp
    train_df = train_df[train_df['plasmid1_ref'] + train_df['plasmid2_ref'] + train_df['plasmid3_ref'] > p_threshold]
    train_df = train_df[train_df['plasmid1_alt'] + train_df['plasmid2_alt'] + train_df['plasmid3_alt'] > p_threshold]
    train_df = train_df[train_df['padj_allele'] < pval]
    train_df['delta'] = train_df['lfc_scaled']
    seqR = []
    seqA = []
    fasta_ref = pysam.FastaFile('reference/hg38.genome.fa')

    for idx, row in train_df.iterrows():
        
        chrom = str(row['chrom'])
        peak_loc = 75
        start = int(row['pos'] - peak_loc) # index? should be right based on SNP matching
        end = int(row['pos'] + peak_loc)
        seq = fasta_ref.fetch(chrom, start, end).upper()
        ref = row['ref']
        alt = row['alt']
        if (idx % 10000 == 0):
            print(ref, alt, seq[75])
        bases = ['A', 'C', 'G', 'T']
        if (ref not in bases or alt not in bases):
            print("ERROR")
            print(ref, alt, seq[75])
            continue
        refseq = insert_variant(seq, ref, peak_loc)
        altseq = insert_variant(seq, alt, peak_loc)
        if(len(seq)!= 150):
            seqR.append("ERROR")
            seqA.append("ERROR")
            continue
        seqR.append(refseq)
        seqA.append(altseq)
    train_df['seq'] = seqR
    train_df['seq2'] = seqA
    train_df = train_df[train_df['seq'] != "ERROR"]
    print(train_df.info())
    print(train_df.head())
    train_df.to_csv(loaded_dir + id + '.csv')