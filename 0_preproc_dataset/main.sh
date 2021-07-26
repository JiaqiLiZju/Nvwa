########################################################################
# wget genome
wget Genome.fasta
wget Annotation.gtf
# promoter
python 0_proc_prom_region_seq.py Genome.fasta Annotation.gtf 500 500 >Species_updown500bp.fa 2>log.Species_updown500bp.fa
# onehot
python 0_onehot_geome.py Species_updown500bp.fa Species_updown500bp.onehot.p

# sc-datasets
# run imputation/pseudocell pipeline
python 1_MAGIC.py
# run 1_gene_label

# dataset
python ../../2_propare_datasets.py train_test_split $Onehot $Label $Annotation Dataset.train_test.h5
python ../../2_propare_datasets.py cross_valid $Onehot $Label $Annotation ./
python ../../2_propare_datasets.py leave_chrom $Onehot $Label $Annotation Dataset.Chrom8_train_test.h5 $GTF 8
python ../../2_propare_datasets.py leave_chrom_CV $Onehot $Label $Annotation ./ $GTF 8


########################################################################
# genome urls
# use repeat-masked genomic fasta

# Human GRCh38
wget ftp://ftp.ensembl.org/pub/release-102/fasta/homo_sapiens/dna//Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz
wget ftp://ftp.ensembl.org/pub/release-102/gtf/homo_sapiens//Homo_sapiens.GRCh38.102.gtf.gz

python 0_proc_prom_region_seq_official.py database_genome/Human_Homo_sapiens/Homo_sapiens.GRCh38.fa database_genome/Human_Homo_sapiens/Homo_sapiens.GRCh38.gtf 10000 10000 >Human_updown10k_official.fa 2>log.Human_updown10k_official.fa
python ./0_onehot_geome.py Human_updown10k_official.fa Human_updown10k_official.onehot.p

# Mouse GRCm38
wget ftp://ftp.ensembl.org/pub/release-88/fasta/mus_musculus//dna/Mus_musculus.GRCm38.dna_sm.toplevel.fa.gz
wget ftp://ftp.ensembl.org/pub/release-88/gtf/mus_musculus//Mus_musculus.GRCm38.102.gtf.gz

# Zebrafish


# Fly / Dmel
# ftp://ftp.flybase.net/genomes//Drosophila_melanogaster/dmel_r6.06_FB2015_03/fasta/dmel-all-chromosome-r6.06.fasta.gz
wget ftp://ftp.flybase.net/genomes//Drosophila_melanogaster/dmel_r6.06_FB2015_03/dna/dmel-raw_scaffolds-r6.06.tar.gz
wget ftp://ftp.flybase.net/genomes//Drosophila_melanogaster/dmel_r6.06_FB2015_03/gtf/dmel-all-r6.06.gtf.gz

# Celegan
wget ftp://ftp.ensembl.org/pub/release-98/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna_sm.toplevel.fa.gz
wget ftp://ftp.ensembl.org/pub/release-98/gtf/caenorhabditis_elegans//Caenorhabditis_elegans.WBcel235.98.gtf.gz

# Smed


########################################################################
# single cell Label
python 1_MAGIC.py
python 1_MAGIC_toLabel.py


########################################################################
# make dataset
# Human
Onehot=../onehot/Human_updown10k.rm_official.onehot.p
GTF=../onehot/Human.gtf_annotation.csv
Label=./HCL_MAGIC_merge_breast_testis_500gene_009.p
Annotation=../HCL_microwell_twotissue_preEmbryo.cellatlas.annotation.txt

python ../../2_propare_datasets.py train_test_split $Onehot $Label $Annotation Dataset.Human_train_test.h5
python ../../2_propare_datasets.py cross_valid $Onehot $Label $Annotation ./
python ../../2_propare_datasets.py leave_chrom $Onehot $Label $Annotation Dataset.Human_Chrom8_train_test.h5 $GTF 8
python ../../2_propare_datasets.py leave_chrom_CV $Onehot $Label $Annotation ./ $GTF 8

# Mouse

# Zebrafish

# Dmel
python ../../2_propare_datasets.py train_test_split ../onehot/Dmel_updown10k.rm_official.onehot.p ./Dmel_MAGIC_01.p ../Drosophila_Microwell_EmbryoStage6_GSE95025.cellatlas.annotation.txt Dataset.Dmel_MAGIC_train_test.h5
python ../../2_propare_datasets.py cross_valid ../onehot/Dmel_updown10k.rm_official.onehot.p ./Dmel_embryo_MAGIC_05.p ../Drosophila_Microwell_EmbryoStage6_GSE95025.cellatlas.annotation.txt ./
python ../../2_propare_datasets.py leave_chrom ../onehot/Dmel_updown10k.rm_official.onehot.p ./Dmel_embryo_MAGIC_05.p ../Drosophila_Microwell_EmbryoStage6_GSE95025.cellatlas.annotation.txt ./ ../onehot/Dmel.gtf_annotation.csv 2R

# Celegan
python ../../2_propare_datasets.py train_test_split ../onehot/Celegan_updown10k.rm_official.onehot.p ./Celegans_005.p ../Celegans_Cao2017.Embryo_GSE126954.cellatlas.annotation.20201213.txt Dataset.Celegans_train_test.h5 &
python ../../2_propare_datasets.py cross_valid ../onehot/Celegan_updown10k.rm_official.onehot.p ./Celegans_005.p ../Celegans_Cao2017.Embryo_GSE126954.cellatlas.annotation.txt ./ &
python ../../2_propare_datasets.py leave_chrom ../onehot/Celegan_updown10k.rm_official.onehot.p ./Celegans_005.p ../Celegans_Cao2017.Embryo_GSE126954.cellatlas.annotation.txt ./ ../onehot/Celegan.gtf_annotation.csv III &

python ../../2_propare_datasets.py train_test_split ../onehot/Celegan_updown10k.rm_official.onehot.p ./Celegans_Embryo_01.p ../Celegans_Cao2017.Embryo_GSE126954.cellatlas.annotation.txt Dataset.Celegans_train_test.h5 &
python ../../2_propare_datasets.py cross_valid ../onehot/Celegan_updown10k.rm_official.onehot.p ./Celegans_Embryo_01.p ../Celegans_Cao2017.Embryo_GSE126954.cellatlas.annotation.txt ./ &
python ../../2_propare_datasets.py leave_chrom ../onehot/Celegan_updown10k.rm_official.onehot.p ./Celegans_Embryo_01.p ../Celegans_Cao2017.Embryo_GSE126954.cellatlas.annotation.txt ./ ../onehot/Celegan.gtf_annotation.csv III &

# Yeast
Onehot=../../PreprocGenome/Yeast_updown10k.official.onehot.p
GTF=../../PreprocGenome/yeast_gtf_annotation.csv
Label=./Yeast.stress.expression_exp.p
Annotation=./Yeast.stress.annotation.txt

python ../2_propare_datasets.py train_test_split $Onehot $Label $Annotation Dataset.train_test_NormLogScale.h5 &
python ../2_propare_datasets.py leave_chrom $Onehot $Label $Annotation Dataset.ChromIV_train_test_NormLogScale.h5 $GTF IV &
python ../2_propare_datasets.py cross_valid $Onehot $Label $Annotation ./ &
python ../2_propare_datasets.py leave_chrom_CV $Onehot $Label $Annotation ./ $GTF IV &
