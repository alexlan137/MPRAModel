import numpy as np
import pandas as pd
import pysam
from mseqgen.sequtils import one_hot_encode

#one-hot encode sequences

def insert_variant(seq, allele, position):
    """ Inserts the specified allele at the given position of the sequence
    """
    left, right = seq[:position-1], seq[position:]
    return left + allele + right

def load_sequences(peaks_df):
    """ Loads the sequences associated with each row of the peaks_df Pandas dataframe
        using chromosome, start position, end position, and allele information
        Returns a list of sequences queried from reference/hg38.genome.fa file
    """
    fasta_ref = pysam.FastaFile('reference/hg38.genome.fa')
    seqref = []
    seqalt = []
    sequences = []
    for idx, row in peaks_df.iterrows():
        start = row['pos'] - (2114 // 2)
        end = row['pos'] + (2114 // 2)
        peak_loc = 1057
        ref = row['ref']
        alt = row['alt']

        seq = fasta_ref.fetch(row['chrom'], start, end).upper()
        sref = insert_variant(seq, ref, peak_loc)
        salt = insert_variant(seq, alt, peak_loc)
        seqref.append(sref)
        seqalt.append(salt)

    Xref = one_hot_encode(seqref, 2114)
    Xalt = one_hot_encode(seqalt, 2114)
    return np.array(Xref), np.array(Xalt), np.array(seqref), np.array(seqalt)

def load_data(filepath):
    data_df = pd.read_csv(filepath)
    peaks_df = data_df[['chrom', 'pos', 'ref', 'alt']]
    score_df = data_df[['delta']]
    XR, XA, seqR, seqA = load_sequences(peaks_df)
    scores = np.array(score_df['delta'])
    return XR, XA, seqR, seqA, scores

def load_data_kampman(filepath):
    PLASMID_ORIG_U = "agaatgaacaagaattattggaattagataaatgggcaagtttgtggaattggtttaacataacaaattggctgtggtatataaaattattcataatgatagtaggaggcttggtaggtttaagaatagtttttgctgtactttctatagtgaatagagttaggcagggatattcaccattatcgtttcagacccacctcccaaccccgaggggacccgacaggcccgaaggaatagaagaagaaggtggagagagagacagagacagatccattcgattagtgaacggatcggcactgcgtgcgccaattctgcagacaaatggcagtattcatccacaattttaaaagaaaaggggggattggggggtacagtgcaggggaaagaatagtagacataatagcaacagacatacaaactaaagaattacaaaaacaaattacaaaaattcaaaattttcgggtttattacagggacagcagagatccagtttggttagtaccgggcccggtgctttgctctgagccagcccaccagtttggaatgactcctttttatgacttgaattttcaagtataaagtctagtgctaaatttaatttgaacaactgtatagtttttgctggttgggggaaggaaaaaaaatggtggcagtgtttttttcagaattagaagtgaaatgaaaacttgttgtgtgtgaggatttctaatgacatgtggtggttgcatactgagtgaagccggtgagcattctgccatgtcaccccctcgtgctcagtaatgtactttacagaaatcctaaactcaaaagattgatataaaccatgcttcttgtgtatatccggtctcttctctgggtagtctcactcagcctgcatttctgccagggcccgctctagacctgcagg"
    PLASMID_UPSTREAM = PLASMID_ORIG_U.upper()

    PLASMID_ORIG_D = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTGCAAAGTGAACACATCGCTAAGCGAAAGCTAAGNNNNNNNNNNNNNNNAccggtcgccaccatggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaggaattcgtcgagggacctaataacttcgtatagcatacattatacgaagttatacatgtttaagggttccgg"
    PLASMID_DOWNSTREAM = PLASMID_ORIG_D.upper()

    data_df = pd.read_csv(filepath)
    peaks_df = data_df[['seq', 'seq2']]
    
    seqref = []
    seqalt = []
    
    for idx, row in peaks_df.iterrows():
        seq1 = PLASMID_UPSTREAM + row['seq'] + PLASMID_DOWNSTREAM
        seq2 = PLASMID_UPSTREAM + row['seq2'] + PLASMID_DOWNSTREAM
        seqalt.append(seq1)
        seqref.append(seq2)
    
    Xref = one_hot_encode(seqref, 2114)
    Xalt = one_hot_encode(seqalt, 2114)
    
    score_df = data_df[['delta']]
    scores = np.array(score_df['delta'])

    return np.array(Xref), np.array(Xalt), np.array(seqref), np.array(seqalt), scores

# if ('__main__'):
#     load_data('data/MPRA/train-abell-filtered.csv')
