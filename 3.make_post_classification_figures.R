######################################################
# 4.make_post_classification_figures.R
# created on March 18 2020
# lucile.neyton@ucsf.edu

# This script aims at generating figures
#   summarising the classifiers built

# Input files (data/metadata folder):
# - CSV-formatted file containing normalised gene counts
# - CSV-formatted file containing mappings between gene symbols and identifiers (cnt_data)
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

# set paths
data_path <- "/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/data/"
results_path <-
  "/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/results/"

#########################
# DATA LOADING - XGB
#########################
algo <- "50000_40_FALSE"
vsd <- read.csv(paste(data_path, "processed/50000_40_FALSE_vsd.csv", sep = ""), 
                row.names = 1)
mapping_data <- read.csv(paste(data_path, "processed/50000_40_FALSE_cnts.csv", sep = ""),
                         row.names = 1)
meta_data <- read.csv(paste(data_path, "processed/50000_40_FALSE_metadata_1vs4.csv", sep = ""),
                      row.names = 1)

best_vars <- read.csv(paste(results_path, "50000_40_FALSE_best_vars_xgb_de_genes.csv", sep = ""),
                         header = FALSE)

#########################
# DATA PREPROCESSING - XGB
#########################
# drop genes with no matching symbol
mapping_data <- mapping_data[, !grepl("EARLI_", colnames(mapping_data), fixed = TRUE), drop = F]
mapping_data <- mapping_data[!is.na(mapping_data$hgnc_symbol), , drop = F]
mapping_data <- mapping_data[mapping_data$hgnc_symbol != "", , drop = F]

# extract symbols 
best_vars_symbols <- c()
for (gene_ in best_vars$V1){
  best_vars_symbols[rownames(mapping_data[mapping_data$hgnc_symbol==gene_, ,drop=F])] <- gene_
}

# add barcodes as row names
rownames(meta_data) <- meta_data$SampleID

# use same data as the one used for the classifier
vst_scaled_data <- data.frame(t(scale(t(vsd))))

#########################
# PLOT COUNTS FOR CLASSIFIER GENES - XGB
#########################
# extract VST counts for selected genes
d <- vst_scaled_data[names(best_vars_symbols), ]
rownames(d) <- best_vars_symbols[rownames(d)]

# melt data frame and add sepsis category
d_melt <- melt(as.matrix(d))
d_melt$sepsis_cat <- meta_data[d_melt$Var2, "sepsis_cat"]

pdf(paste(results_path, paste(algo, "profile_plot.pdf", sep = "_"), sep = ""),
    width = 8, height = 5)
ggplot(d_melt, aes(x=Var1, y=value, group=Var2, color=sepsis_cat)) +
  geom_line()
dev.off()

# heatmap
annot_data <- meta_data[, "sepsis_cat", drop = F]
annot_data$sepsis_cat <- as.factor(annot_data$sepsis_cat)
annot_data$sepsis_cat <- relevel(annot_data$sepsis_cat, ref="4_NO_Sepsis")

pdf(paste(results_path, paste(algo, "hm.pdf", sep = "_"), sep = ""),
    width = 8, height = 5)
pheatmap(d, annotation_col = annot_data, scale = "none",
         show_colnames = FALSE, cutree_cols = 2)
dev.off()

#########################
# DATA LOADING - BSVM
#########################
algo <- "50000_20_FALSE"
vsd <- read.csv(paste(data_path, "processed/50000_20_FALSE_vsd.csv", sep = ""), 
                row.names = 1)
mapping_data <- read.csv(paste(data_path, "processed/50000_20_FALSE_cnts.csv", sep = ""),
                         row.names = 1)
meta_data <- read.csv(paste(data_path, "processed/50000_20_FALSE_metadata_1vs4.csv", sep = ""),
                      row.names = 1)

best_vars <- read.csv(paste(results_path, "50000_20_FALSE_best_vars_bsvm_de_genes.csv", sep = ""),
                          header = FALSE)

#########################
# DATA PREPROCESSING - BSVM
#########################
# drop genes with no matching symbol
mapping_data <- mapping_data[, !grepl("EARLI_", colnames(mapping_data), fixed = TRUE), drop = F]
mapping_data <- mapping_data[!is.na(mapping_data$hgnc_symbol), , drop = F]
mapping_data <- mapping_data[mapping_data$hgnc_symbol != "", , drop = F]

# extract symbols 
best_vars_symbols <- c()
for (gene_ in best_vars$V1){
  best_vars_symbols[rownames(mapping_data[mapping_data$hgnc_symbol==gene_, ,drop=F])] <- gene_
}

# add barcodes as row names
rownames(meta_data) <- meta_data$SampleID

# use same data as the one used for the classifier
vst_scaled_data <- data.frame(t(scale(t(vsd))))

#########################
# PLOT COUNTS FOR CLASSIFIER GENES - BSVM
#########################
# extract VST counts for selected genes
d <- vst_scaled_data[names(best_vars_symbols), ]
rownames(d) <- best_vars_symbols[rownames(d)]

# melt data frame and add sepsis category
d_melt <- melt(as.matrix(d))
d_melt$sepsis_cat <- meta_data[d_melt$Var2, "sepsis_cat"]

pdf(paste(results_path, paste(algo, "profile_plot.pdf", sep = "_"), sep = ""),
    width = 8, height = 5)
ggplot(d_melt, aes(x=Var1, y=value, group=Var2, color=sepsis_cat)) +
  geom_line()
dev.off()

# heatmap
annot_data <- meta_data[, "sepsis_cat", drop = F]
annot_data$sepsis_cat <- as.factor(annot_data$sepsis_cat)
annot_data$sepsis_cat <- relevel(annot_data$sepsis_cat, ref="4_NO_Sepsis")

pdf(paste(results_path, paste(algo, "hm.pdf", sep = "_"), sep = ""),
    width = 8, height = 5)
pheatmap(d, annotation_col = annot_data, scale = "none",
         show_colnames = FALSE, cutree_cols = 2)
dev.off()

