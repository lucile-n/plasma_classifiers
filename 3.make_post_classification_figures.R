######################################################
# 3.make_post_classification_figures.R
# created on May 6 2021
# lucile.neyton@ucsf.edu

# This script aims at generating figures
#   summarising the classifiers built (only the full training fold)

# Input files (data folder):
# - CSV-formatted file containing normalised gene counts
# - CSV-formatted file containing mappings between gene symbols and identifiers
# - CSV-formatted file containing metadata
# - A CSV-formatted file containing selected variables

# Output files (results folder):
# - One PDF file for selected genes showing normalised values per label
# - A PDF file representing a heatmap of normalised values for the same genes
######################################################

rm(list = ls())
setwd("/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/")

# load libraries
library(ggplot2) # 3.3.2
library(pheatmap) # 1.0.12
library(reshape2) # 1.4.4

# set paths
data_path <-
  "/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/data/"
alt_data_path <-
  "/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_VALID/data/"
results_path <-
  "/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/results/"

# get meta data
meta_data_path <- paste(alt_data_path, paste("processed/", "EARLI_metadata_adjudication_IDseq_LPSstudyData_7.5.20_viruspos.csv", sep = ""), sep = "")
meta_data <- read.csv(meta_data_path)

# add barcodes as row names
rownames(meta_data) <- sapply(meta_data$Barcode, function(x) paste("EARLI", x, sep = "_"))

# change group name
colnames(meta_data)[colnames(meta_data) == "Group"] <- "sepsis_cat"

# list models to visualise
algos_ <- c(
  "plasma_50000_20_0.1_plasma_TRUE_xgb_1vs4_sepsis", "paxgene_50000_20_0.1_paxgene_TRUE_xgb_1vs4_sepsis",
  "plasma_50000_20_0.1_plasma_TRUE_bsvm_1vs4_sepsis", "paxgene_50000_20_0.1_paxgene_TRUE_bsvm_1vs4_sepsis",
  "plasma_50000_20_0.1_plasma_TRUE_xgb_12vs4_sepsis", "paxgene_50000_20_0.1_paxgene_TRUE_xgb_12vs4_sepsis",
  "plasma_50000_20_0.1_plasma_TRUE_bsvm_12vs4_sepsis", "paxgene_50000_20_0.1_paxgene_TRUE_bsvm_12vs4_sepsis",
  "plasma_50000_20_0.1_plasma_TRUE_bsvm_12_virus", "paxgene_50000_20_0.1_paxgene_TRUE_bsvm_12_virus",
  "plasma_50000_20_0.1_plasma_TRUE_bsvm_124_virus", "paxgene_50000_20_0.1_paxgene_TRUE_bsvm_124_virus"
)

# "overlap" or "all_genes"
genes_to_use <- "overlap"

# save outputs to specific folders
results_path <- paste(results_path, paste(genes_to_use, "/", sep=""), sep="")
data_path <- paste(data_path, paste(genes_to_use, "/", sep=""), sep="")

for (i in c(1:length(algos_))) {
  # extract paramaters
  algo_ <- algos_[i]
  split_prefix <- strsplit(algo_, "_")[[1]]
  target_cat <- split_prefix[9]
  file_prefix_tmp <- paste(split_prefix[2], paste(split_prefix[3], paste(split_prefix[4], paste(split_prefix[6], paste(split_prefix[8], target_cat, sep="_"), sep = "_"), sep = "_"), sep = "_"), sep = "_")

  # generate input file and output prefixes
  file_prefix <- paste(file_prefix_tmp, split_prefix[1], sep = "_")

  output_prefix <- algo_
  
  #########################
  # DATA LOADING
  #########################
  # load data
  vsd_path <- paste(data_path, paste("processed/", paste(file_prefix, "_vsd.csv", sep = ""), sep = ""), sep = "")
  vsd <- read.csv(vsd_path, row.names = 1, check.names = F)

  mapping_data_path <- paste(data_path, paste("processed/", paste(file_prefix, "_cnts.csv", sep = ""), sep = ""), sep = "")
  mapping_data <- read.csv(mapping_data_path, row.names = 1)

  best_vars_path <- paste(results_path, paste(algo_, "_full_best_vars.csv", sep = ""), sep = "")
  best_vars <- read.csv(best_vars_path, header = FALSE)

  #########################
  # DATA PRE-PROCESSING
  #########################
  # drop genes with no matching symbol
  mapping_data <- mapping_data[, !grepl("EARLI_", colnames(mapping_data), fixed = TRUE), drop = F]
  mapping_data <- mapping_data[!is.na(mapping_data$hgnc_symbol), , drop = F]
  mapping_data <- mapping_data[mapping_data$hgnc_symbol != "", , drop = F]

  # extract symbols
  best_vars_symbols <- c()
  for (gene_ in best_vars$V1) {
    if (!is.na(mapping_data[gene_, ])){
      best_vars_symbols[gene_] <- mapping_data[gene_, ]
    }else{
      best_vars_symbols[gene_] <- gene_
    }
    
  }

  # extract VST counts for selected genes
  d <- vsd[names(best_vars_symbols), ]

  # row scaling for vis
  vst_scaled_data <- t(scale(t(d)))

  # add gene symbols
  rownames(vst_scaled_data) <- best_vars_symbols[rownames(vst_scaled_data)]

  #########################
  # DATA VIS
  #########################
  # plot counts
  # melt data frame and add sepsis category
  d_melt <- melt(as.matrix(vst_scaled_data))

  if (target_cat == "sepsis"){
    target_var <- "sepsis_cat"
    ref_level <- "4_NO_Sepsis"
    d_melt$target_cat <- meta_data[as.character(d_melt$Var2), target_var]
  }else{
    if (target_cat == "virus"){
      target_var <- "viruspos"
      ref_level <- "nonviral"
      d_melt$target_cat <- meta_data[as.character(d_melt$Var2), target_var]
    }
  }

  # profile plot
  pdf(paste(results_path, paste(output_prefix, "profile_plot.pdf", sep = "_"), sep = ""),
    width = 15, height = 9
  )
  print(ggplot(d_melt, aes(x = Var1, y = value, group = Var2, color = target_cat)) +
    geom_line() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)))
  dev.off()

  # heatmap
  annot_data <- meta_data[colnames(vst_scaled_data), target_var, drop = F]
  colnames(annot_data) <- "target_cat"
  annot_data$target_cat <- as.factor(annot_data$target_cat)
  annot_data$target_cat <- relevel(annot_data$target_cat, ref = ref_level)

  pdf(paste(results_path, paste(output_prefix, "hm.pdf", sep = "_"), sep = ""),
    width = 10, height = 10
  )
  print(pheatmap(vst_scaled_data,
    annotation_col = annot_data, scale = "none",
    show_colnames = FALSE, cutree_cols = 2
  ))
  dev.off()
}
