import numpy as np
import pandas as pd
import pysam

def insert_variant(seq, allele, position):
    """ Inserts the specified allele at the given position of the sequence
    """
    left, right = seq[:position-1], seq[position:]
    return left + allele + right

def loadQTL():
    raw_df = pd.read_csv('data/QTL/AF.all.ATAC.MKK.1000G.merge.fs.IDs.nm.ba.sorted.asb_summary.permuted_caqtl.clean.txt', delimiter='\t')
    print(raw_df.info())
    print(raw_df.head(10))
    filtered_df = raw_df[raw_df['mean_signif_altCount'] != '.']
    filtered_df = filtered_df[filtered_df['mean_signif_refCount'] != '.']
    filtered_df = filtered_df[filtered_df['mean_signif_altCount'] != '']
    filtered_df = filtered_df[filtered_df['mean_signif_refCount'] != '']
    
    
    key_df = filtered_df[['chr', 'start', 'stop', 'ref', 'alt', 'variant', 'mean_signif_refCount', 'mean_signif_altCount']]
    key_df = key_df[key_df['mean_signif_refCount'].notnull()]
    print(key_df.head(10))
    print(key_df.info())

    fasta_ref = pysam.FastaFile('reference/hg38.genome.fa')
    key_df['lfc'] = np.log2(key_df['mean_signif_refCount'].astype(float) / key_df['mean_signif_altCount'].astype(float))
    seqR = []
    seqA = []
    for idx, row in key_df.iterrows():
        if (idx % 10 != 0):
            seqR.append("RM")
            seqA.append("RM")
            continue
        chrom = 'chr' + str(row['chr'])
        peak_loc = 1057
        start = int(row['start'] - peak_loc)
        end = int(row['start'] + peak_loc)
        seq = fasta_ref.fetch(chrom, start, end).upper()
        ref = row['ref']
        alt = row['alt']
        if (idx % 10000 == 0):
            print(ref, alt, seq[1057])
        bases = ['A', 'C', 'G', 'T']
        if (ref not in bases or alt not in bases):
            print("ERROR", ref, alt, seq[1057])
            continue
        refseq = insert_variant(seq, ref, peak_loc)
        altseq = insert_variant(seq, alt, peak_loc)
        if(len(seq)!= 2114):
            print("ERROR_LEN")
            continue
        seqR.append(refseq)
        seqA.append(altseq)
    key_df['seq'] = seqR
    key_df['seq2'] = seqA
    key_df = key_df[key_df['seq'] != "RM"]
    key_df.to_csv('data/QTL/1000GV2tenth.csv')

def filterKeyDF():
    key_df = pd.read_csv('data/QTL/key_df_signif.csv')
    key_df = key_df[key_df['mean_signif_refCount'].notnull()]
    key_df.to_csv('data/QTL/key_df_filtered.csv')

if ('__main__'):
    loadQTL()