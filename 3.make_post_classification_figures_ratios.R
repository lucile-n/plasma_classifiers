######################################################
# 3.make_post_classification_figures_ratios.R
# created on March 31 2021
# lucile.neyton@ucsf.edu

# This script aims at generating figures
#   summarising the classifiers built using gene ratios

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
algo <- "50000_40_FALSE_ratios"
vsd <- read.csv(paste(data_path, "processed/50000_30_0.1_TRUE_vsd.csv", sep = ""), 
                row.names = 1)
mapping_data <- read.csv(paste(data_path, "processed/50000_30_0.1_TRUE_cnts.csv", sep = ""),
                         row.names = 1)
meta_data <- read.csv(paste(data_path, "processed/50000_30_0.1_TRUE_metadata_1vs4.csv", sep = ""),
                      row.names = 1)

best_vars <- read.csv(paste(results_path, "50000_30_0.1_TRUE_ratios_best_vars_xgb_de_genes.csv", sep = ""),
                         header = FALSE)

#########################
# DATA PREPROCESSING - XGB
#########################
# drop genes with no matching symbol
mapping_data <- mapping_data[, !grepl("EARLI_", colnames(mapping_data), fixed = TRUE), drop = F]
mapping_data <- mapping_data[!is.na(mapping_data$hgnc_symbol), , drop = F]
mapping_data <- mapping_data[mapping_data$hgnc_symbol != "", , drop = F]

# extract symbols 
best_vars_split <- unique(unlist(sapply(best_vars$V1, function(x) strsplit(x, "/", fixed = TRUE))))

best_vars_symbols <- c()
for (gene_ in best_vars_split){
  best_vars_symbols[rownames(mapping_data[mapping_data$hgnc_symbol==gene_, ,drop=F])] <- gene_
}

# add barcodes as row names
rownames(meta_data) <- meta_data$SampleID

#########################
# PLOT COUNTS FOR CLASSIFIER GENES - XGB
#########################
# extract VST counts for selected genes
d <- vsd[names(best_vars_symbols), ]
rownames(d) <- best_vars_symbols[rownames(d)]

# compute ratios
gene_ratio_df <- matrix(nrow = length(best_vars$V1), ncol = length(colnames(d)))
for (gene_ratio in 1:length(best_vars$V1)){
  gene_1 <- strsplit(best_vars$V1[gene_ratio], "/", fixed = TRUE)[[1]][1]
  gene_2 <- strsplit(best_vars$V1[gene_ratio], "/", fixed = TRUE)[[1]][2]
  gene_ratio_df[gene_ratio, ] <- unlist(unname(d[gene_1, ] / d[gene_2, ]))
}
gene_ratio_df <- data.frame(gene_ratio_df)
colnames(gene_ratio_df) <- colnames(d)
rownames(gene_ratio_df) <- best_vars$V1

# melt data frame and add sepsis category
d_melt <- melt(as.matrix(gene_ratio_df))
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
pheatmap(gene_ratio_df, annotation_col = annot_data, scale = "none",
         show_colnames = FALSE, cutree_cols = 2)
dev.off()
