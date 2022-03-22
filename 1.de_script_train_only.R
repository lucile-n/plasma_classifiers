######################################################
# de_script_train_only.R
# created on November 3, 2021
# lucile.neyton@ucsf.edu

# This script aims at performing a differential
#   expression analysis between samples,
#   given sepsis/viral status, and using only samples
#   from the training set

# Input files (data folder):
# - CSV-formatted file containing gene counts from PAXgene and plasma samples
#   (genes x samples)
# - CSV-formatted files containing sepsis/viral status
#   and covariates (samples x label)

# Output files (results folder):
# - CSV file listing significantly differentially expressed genes
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
# SET PARAMETERS
#########################
# all parameter values to test
# minimum number of counts per sample in plasma
min_cnts_per_sample_vals <- c(50000)
# percentage of samples with non-zero counts per gene
min_non_zero_counts_per_genes_vals <- c(20)
# FDR threshold to call a gene differentially expressed
fdr_thresh_vals <- c(0.5)
nominal_pval <- F
# target variable
targets_vals <- c("sepsis", "virus")

# generate a matrix with all possible combinations of the above parameters
comb_mat <- expand.grid(
  "min_cnts_per_sample" = min_cnts_per_sample_vals,
  "min_non_zero_counts_per_genes" = min_non_zero_counts_per_genes_vals,
  "fdr_thresh" = fdr_thresh_vals,
  "target_val" = targets_vals
)

comb_mat$test_probs_files <- c(paste0(results_path, "all_genes/plasma_50000_20_0.1_plasma_TRUE_bsvm_12vs4_sepsis_full_test_probs.csv"),
                               paste0(results_path, "all_genes/plasma_50000_20_0.1_plasma_TRUE_bsvm_12_virus_full_test_probs.csv"))

#########################
# DATA LOADING
#########################
# list data paths
# plasma counts
plasma_data_path <- paste(data_path, "raw/earli_plasma_counts.csv", sep = "")
# paxgene counts
paxgene_data_path <- paste(data_path, "raw/EARLI_star_pc_and_lincRNA_genecounts.qc.tsv", sep = "")
# meta data
metadata_path <- paste(extra_data_path, "processed/EARLI_metadata_adjudication_IDseq_LPSstudyData_7.5.20.csv", sep = "")
# virus status
virus_status_data_path <- paste(data_path, "raw/viruspos_samples.txt", sep = "")

# read data in
plasma_data <- read.csv(plasma_data_path, row.names = 1)
paxgene_data <- read.table(paxgene_data_path, row.names = 1, sep = "\t")
meta_data <- read.csv(metadata_path)
virus_status_data <- read.table(virus_status_data_path)

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

# add virus status to metadata frame
virus_vect <- c()
for (sample_id in meta_data$EARLI_Barcode) {
  if (!is.na(sample_id)) {
    if (sample_id %in% virus_status_data$V1) {
      virus_vect <- c(virus_vect, "viral")
    } else {
      virus_vect <- c(virus_vect, "nonviral")
    }
  } else {
    virus_vect <- c(virus_vect, NA)
  }
}
meta_data$viruspos <- virus_vect

# filter count data to only keep samples in both the plasma and PAXgene datasets
samples_in_common <- intersect(colnames(plasma_data), colnames(paxgene_data))
plasma_data <- plasma_data[, colnames(plasma_data) %in% samples_in_common]
paxgene_data <- paxgene_data[, colnames(paxgene_data) %in% samples_in_common]

# all ENSG genes
all_ensg_genes <- c(union(rownames(plasma_data), rownames(paxgene_data)))

# 50,000 filter
# filter plasma samples with less than N protein coding genes-associated counts
sample_names <- colnames(plasma_data)
samples_to_drop <- sample_names[(plasma_data[1, ] <= 50000)]

plasma_data_filt <- plasma_data[, !(colnames(plasma_data) %in% samples_to_drop)]

# for each parameters combination
for (row_ in rownames(comb_mat)) {
  # extract parameter values
  min_cnts_per_sample <- comb_mat[row_, "min_cnts_per_sample"]
  min_non_zero_counts_per_genes <- comb_mat[row_, "min_non_zero_counts_per_genes"]
  fdr_thresh <- comb_mat[row_, "fdr_thresh"]
  target_val <- comb_mat[row_, "target_val"]
  
  # set two options for sepsis/virus classifiers
  if (target_val=="sepsis"){
    comp_ <- "12vs4"
  }else{
    if (target_val=="virus"){
      comp_ <- "12"
    }
  }
    
  # generate unique prefix for that combination
  if (nominal_pval){
    results_prefix <- paste(as.character(as.integer(min_cnts_per_sample)),
                            paste(paste(min_non_zero_counts_per_genes, paste0(fdr_thresh, "_nominalp"), sep = "_"), 
                                  paste(comp_, target_val, sep="_"), sep = "_"),
                            sep = "_")
  }else{
    results_prefix <- paste(as.character(as.integer(min_cnts_per_sample)),
                            paste(paste(min_non_zero_counts_per_genes, fdr_thresh, sep = "_"), 
                                  paste(comp_, target_val, sep="_"), sep = "_"),
                            sep = "_")
  }
  
                          
  # read test predictions
  test_probs_file <- comb_mat[row_, "test_probs_files"]
  test_probs <- read.csv(test_probs_file)
  test_samples_to_exclude <- test_probs$X

  #########################
  # DATA PREPROCESSING
  #########################
  # filter plasma samples with less than N protein coding genes-associated counts
  sample_names <- colnames(plasma_data)
  samples_to_drop <- sample_names[(plasma_data[1, ] <= min_cnts_per_sample)]

  plasma_data_filt <- plasma_data[, !(colnames(plasma_data) %in% samples_to_drop)]

  # keep only the plasma samples of interest depending on the comparison we want to make
  if ((comp_ == "12vs4")) {
    selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+", "4_NO_Sepsis"), "EARLI_Barcode"]
  }else{
    if (comp_ == "12"){
      selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+"), "EARLI_Barcode"]
    }
  }
    
  plasma_data_filt <- plasma_data_filt[, colnames(plasma_data_filt) %in% c(selected_samples)]
  all_samples <- colnames(plasma_data_filt)
  
  # exclude test samples
  plasma_data_filt <- plasma_data_filt[, !(colnames(plasma_data_filt) %in% test_samples_to_exclude)]
  
  # filter the meta data to match the selected samples
  # make sure our variable of interest is a factor
  meta_data_filt <- meta_data[meta_data$EARLI_Barcode %in% colnames(plasma_data_filt), ]
  meta_data_filt$sepsis_cat <- as.factor(meta_data_filt$sepsis_cat)
  meta_data_filt$viruspos <- as.factor(meta_data_filt$viruspos)
  meta_data_filt <- meta_data_filt[!(meta_data_filt$EARLI_Barcode %in% test_samples_to_exclude), ]

  # filter genes present in only some of the plasma samples (use only samples with more than the minimum count)
  # ignore first row (gene counts per sample) and last column (gene symbols)
  gene_names <- rownames(plasma_data_filt[2:nrow(plasma_data_filt), ])
  gene_data <- plasma_data_filt[2:nrow(plasma_data_filt), ]
  genes_to_drop <- gene_names[apply(gene_data, 1, function(x) sum(x > 0)) / (ncol(plasma_data_filt)) < (min_non_zero_counts_per_genes / 100)]
  plasma_data_filt <- plasma_data_filt[!(rownames(plasma_data_filt) %in% genes_to_drop), ]

  # replace missing age values by average for that subgroup
  if (target_val == "sepsis"){
    target_col <- "sepsis_cat"
  }else{
    if (target_val == "virus"){
      target_col <- "viruspos"
    }
  }
  for (target_val_cat in unique(meta_data_filt[, target_col])) {
    meta_data_filt$Age[is.na(meta_data_filt$Age) & meta_data_filt[, target_col] == target_val_cat] <-
      mean(meta_data_filt$Age[(!is.na(meta_data_filt$Age)) & meta_data_filt[, target_col] == target_val_cat])
  }

  # standard scaling on Age
  meta_data_filt$age_scaled <- scale(meta_data_filt$Age)

  # make sure gender is a factor
  meta_data_filt$gender <- as.factor(meta_data_filt$Gender)

  # drop first row and column and store gene symbols
  plasma_data_filt <- plasma_data_filt[2:nrow(plasma_data_filt), ]
  
  # save filtered counts
  # make sure test samples included
  write.csv(plasma_data[rownames(plasma_data_filt), colnames(plasma_data) %in% c(colnames(plasma_data_filt), test_samples_to_exclude)], 
            paste0(data_path, paste0("all_genes/processed/", paste(results_prefix, "plasma_train_only_cnts.csv", sep = "_"))))
  
  #########################
  # DE ANALYSIS
  #########################
  # build DESeq2 object
  meta_data_tmp <- meta_data_filt
  
  # for the classifiers, we only care about sepsis vs non-sepsis
  meta_data_tmp$sepsis_cat[meta_data_tmp$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+")] <- "1_Sepsis+BldCx+"
  
  meta_data_full <- meta_data[meta_data$EARLI_Barcode %in% all_samples, ]
  meta_data_full$sepsis_cat <- as.factor(meta_data_full$sepsis_cat)
  meta_data_full$viruspos <- as.factor(meta_data_full$viruspos)
  write.csv(meta_data_full, paste(data_path, paste("processed/", paste(results_prefix, "train_only_metadata.csv", sep = "_"), sep = ""), sep = ""))

  # add target val as a single label
  if (target_val == "sepsis"){
    meta_data_tmp$target_val <- meta_data_tmp$sepsis_cat
  }else{
    if (target_val == "virus"){
      meta_data_tmp$target_val <- meta_data_tmp$viruspos
    }
  }
  
  # make sure meta_data has the same order as the counts matrices
  rownames(meta_data_tmp) <- meta_data_tmp$EARLI_Barcode
  meta_data_tmp <- meta_data_tmp[colnames(plasma_data_filt), ]
  
  # if we want to include age and sex
  dds_plasma <- DESeqDataSetFromMatrix(
  	countData = plasma_data_filt,
    colData = meta_data_tmp,
    design = ~ target_val + age_scaled + gender
    )

  # choose the reference level for the factor of interest
  if (target_val == "sepsis"){
    dds_plasma$target_val <- relevel(dds_plasma$target_val, ref = "4_NO_Sepsis")
  }else{
    if (target_val == "virus"){
      dds_plasma$target_val <- relevel(dds_plasma$target_val, ref = "nonviral")
    }
  }

  # run DESeq
  dds_plasma <- DESeq(dds_plasma)

  # extract DESeq2 results
  res_plasma <- lfcShrink(dds_plasma, coef=2, type ="apeglm") 

  # sort the genes from lowest to highest given adjusted p-values
  res_plasma <- res_plasma[order(res_plasma$padj, decreasing = F), ]

  # replace NA values with 1s and keep only significant genes
  res_plasma$padj[is.na(res_plasma$padj)] <- 1
  
  if (nominal_pval){
    res_plasma$pvalue[is.na(res_plasma$pvalue)] <- 1
    sig_results_plasma <- data.frame(res_plasma[res_plasma$pvalue < fdr_thresh, ])
  }else{
    sig_results_plasma <- data.frame(res_plasma[res_plasma$padj < fdr_thresh, ])
  }
  
  # save the output as a CSV file
  write.csv(sig_results_plasma, paste(results_path, paste(results_prefix, "plasma_train_only_DGEA_results.csv", sep = "_"), sep = ""))

  # save unfiltered outputs (for IPA)
  write.csv(data.frame(res_plasma), paste(results_path, paste(results_prefix, "plasma_train_only_DGEA_unfiltered_results.csv", sep = "_"), sep = ""))
}
  
