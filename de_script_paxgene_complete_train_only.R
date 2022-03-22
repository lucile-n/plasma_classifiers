######################################################
# de_script_paxgene_complete_train_only.R
# created on November 3, 2021
# lucile.neyton@ucsf.edu

# This script aims at performing a differential
#   expression analysis between samples,
#   given sepsis/viral status, and using only samples
#   from the training set (PAXgene)

# Input files (data folder):
# - CSV-formatted file containing gene counts from PAXgene samples
#   (genes x samples)
# - CSV-formatted files containing sepsis status
#   and Sex and Age (samples x label)

# Output files (results folder):
# - One CSV file with significantly differentially expressed genes per run.
######################################################

rm(list = ls())
setwd("/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/")

# load libraries
library(DESeq2) # 1.28.1

# set paths
data_path <-
  "/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/data/"
extra_data_path <-
  "/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_VALID/data/"
results_path <-
  "/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/results/"

#########################
# DATA LOADING
#########################
# list data paths
# paxgene counts
paxgene_data_path <- paste(data_path, "raw/EARLI_star_pc_and_lincRNA_genecounts.qc.tsv", sep = "")
# meta data
metadata_path <- paste(extra_data_path, "processed/EARLI_metadata_adjudication_IDseq_LPSstudyData_7.5.20_viruspos.csv", sep = "")

# read data in
paxgene_data <- read.table(paxgene_data_path, row.names = 1, sep = "\t")
meta_data <- read.csv(metadata_path)

# reformat PAXgene data
# make sure we only keep samples from the counts matrix that are listed in HOST_PAXgene_filename
paxgene_data <- paxgene_data[, colnames(paxgene_data) %in% meta_data$HOST_PAXgene_filename]
# drop duplicated ENSG identifiers
paxgene_data <- paxgene_data[!grepl(pattern = ".*(\\.).*(\\.).*", rownames(paxgene_data)), ]
rownames(paxgene_data) <- sapply(rownames(paxgene_data), function(x) strsplit(x, ".", fixed = TRUE)[[1]][1])
colnames(paxgene_data) <- sapply(colnames(paxgene_data), function(x) paste(strsplit(x, "_")[[1]][1], strsplit(x, "_")[[1]][2], sep = "_"))

# format meta data
meta_data$EARLI_Barcode <- sapply(meta_data$Barcode, function(x) paste("EARLI", x, sep = "_"))
colnames(meta_data)[colnames(meta_data) == "Group"] <- "sepsis_cat"

# all ENSG genes
all_ensg_genes <- rownames(paxgene_data)

# set two options for sepsis/virus classifiers
target_ <- "virus"
fdr_thresh <- 0.1

if (target_=="virus"){
  comp_ <- "12"
}else{
  if (target_=="sepsis"){
    comp_ <- "12vs4"
  }
}

# generate a unique prefix for that combination
# FDR 0.1 and TRUE (covariates included in model)
results_prefix <- paste(paste(paste(fdr_thresh, "TRUE", sep="_"), comp_, sep="_"), target_, sep="_")

#########################
# DATA PREPROCESSING
#########################
# keep only the plasma samples of interest depending on the comparison we want to make
if (comp_ == "12"){
  selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+"), "EARLI_Barcode"]
}else{
  if (comp_ == "12vs4"){
    selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+", "4_NO_Sepsis"), "EARLI_Barcode"]
  }
}
selected_samples <- selected_samples[selected_samples %in% colnames(paxgene_data)]

# filter the meta data to match the selected samples
# make sure our variable of interest is a factor
meta_data_filt <- meta_data[meta_data$EARLI_Barcode %in% selected_samples, ]
meta_data_filt$viruspos <- as.factor(meta_data_filt$viruspos)
meta_data_filt$sepsis_cat <- as.factor(meta_data_filt$sepsis_cat)

# for paxgene data, filter samples to keep the ones selected + order
paxgene_data_filt <- paxgene_data[, colnames(paxgene_data) %in% c(meta_data_filt$EARLI_Barcode)]
paxgene_data_filt <- paxgene_data_filt[, c(meta_data_filt$EARLI_Barcode)]

# replace missing age values by average for that subgroup
if (target_=="virus"){
  target_col <- "viruspos"
  test_probs_file <- paste0(results_path, "paxgene_complete_0.1_TRUE_12_virus_bsvm_test_probs.csv")
}else{
  if (target_=="sepsis"){
    target_col <- "sepsis_cat"
    test_probs_file <- paste0(results_path, "paxgene_complete_0.1_TRUE_12vs4_sepsis_bsvm_test_probs.csv")
  }
}

# exclude test samples
test_probs <- read.csv(test_probs_file)
test_samples_to_exclude <- test_probs$sample_ids

meta_data_filt <- meta_data_filt[!(meta_data_filt$EARLI_Barcode %in% test_samples_to_exclude), ]
paxgene_data_filt <- paxgene_data_filt[, !(colnames(paxgene_data_filt) %in% test_samples_to_exclude)]

for (target_val_cat in unique(meta_data_filt[, target_col])) {
  meta_data_filt$Age[is.na(meta_data_filt$Age) & meta_data_filt[, target_col] == target_val_cat] <-
    mean(meta_data_filt$Age[(!is.na(meta_data_filt$Age)) & meta_data_filt[, target_col] == target_val_cat])
}

# standard scaling on Age
meta_data_filt$age_scaled <- scale(meta_data_filt$Age)
# make sure gender is a factor
meta_data_filt$gender <- as.factor(meta_data_filt$Gender)

#########################
# DE ANALYSIS
#########################
# build DESeq2 object
meta_data_tmp <- meta_data_filt

# for the classifiers, we only care about sepsis vs non-sepsis
meta_data_tmp$sepsis_cat[meta_data_tmp$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+")] <- "1_Sepsis+BldCx+"

# add target val as a single label
meta_data_tmp$target_val <- meta_data_tmp[, target_col]

# make sure meta_data has the same order as the counts matrices
rownames(meta_data_tmp) <- meta_data_tmp$EARLI_Barcode
meta_data_tmp <- meta_data_tmp[colnames(paxgene_data_filt), ]

# if we want to include age and sex
dds_paxgene <- DESeqDataSetFromMatrix(
    countData = paxgene_data_filt,
    colData = meta_data_tmp,
    design = ~ target_val + age_scaled + gender
)

# choose the reference level for the factor of interest
if (target_=="virus"){
  dds_paxgene$target_val <- relevel(dds_paxgene$target_val, ref = "nonviral")
}else{
  if (target_=="sepsis"){
    dds_paxgene$target_val <- relevel(dds_paxgene$target_val, ref = "4_NO_Sepsis")
  }
}

# run DESeq
dds_paxgene <- DESeq(dds_paxgene)

# extract DESeq2 results
if (target_=="virus"){
  #res_paxgene <- results(dds_paxgene, contrast = c("target_val", "viral", "nonviral"))
  res_paxgene <- lfcShrink(dds_paxgene, coef=2, type ="apeglm")
}else{
  if (target_=="sepsis"){
    #res_paxgene <- results(dds_paxgene, contrast = c("target_val", "1_Sepsis+BldCx+", "4_NO_Sepsis"))
    res_paxgene <- lfcShrink(dds_paxgene, coef=2, type ="apeglm")
  }
}

# sort the genes from lowest to highest given adjusted p-values
res_paxgene <- res_paxgene[order(res_paxgene$padj, decreasing = F), ]

# replace NA values with 1s and keep only significant genes
res_paxgene$padj[is.na(res_paxgene$padj)] <- 1
sig_results_paxgene <- data.frame(res_paxgene[res_paxgene$padj < fdr_thresh, ])

# save the output as a CSV file
write.csv(sig_results_paxgene, paste(results_path, paste(results_prefix, "paxgene_complete_train_only_DGEA_results.csv", sep = "_"), sep = ""))

# save the complete list (for IPA)
write.csv(data.frame(res_paxgene), paste(results_path, paste(results_prefix, "paxgene_complete_train_only_DGEA_unfiltered_results.csv", sep = "_"), sep = ""))

