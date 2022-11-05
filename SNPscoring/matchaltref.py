import numpy as np
import pandas as pd
import pysam

SNPs = pd.read_csv('data/GWAS/Bellenguez/ShreyaSNPsT10.csv')
print(SNPs.info())
fasta_ref = pysam.FastaFile('reference/hg38.genome.fa')
ref = []
alt = []
for idx, row in SNPs.iterrows():
    base = fasta_ref.fetch(row['chrom'], int(row['pos']), int(row['pos']) + 1).upper()
    print(row['AlleleA'], row['AlleleB'], base)
    if(row['AlleleA'] == base):
        ref.append(row['AlleleA'])
        alt.append(row['AlleleB'])
    elif(row['AlleleB'] == base):
        ref.append(row['AlleleB'])
        alt.append(row['AlleleA'])

SNPs['ref'] = ref
SNPs['alt'] = alt

SNPs.to_csv('data/GWAS/Bellenguez/ShreyaSNPsT10refalt.csv')
    
