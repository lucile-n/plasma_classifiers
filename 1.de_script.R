######################################################
# 1.de_script.R
# created on March 15 2021
# lucile.neyton@ucsf.edu

# This script aims at performing a differential
#   expression analysis between samples,
#   given their sepsis/viral status

# Input files (data folder):
# - CSV-formatted file containing gene counts from PAXgene and plasma samples
#   (genes x samples)
# - CSV-formatted files containing sepsis/viral status
#   and Sex and Age (samples x label)

# Output files (results folder):
# - One file with counts values per run.
# - One file with normalised (unscaled) gene expression values per run.
# - One metadata file per run.
# - One CSV file with significantly differentially expressed genes per run.
######################################################

rm(list = ls())
setwd("/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/")

# load libraries
library(DESeq2) # 1.28.1
library(readxl) # 1.3.1
library(ggfortify) # 0.4.11
library(ggrepel) # 0.8.2
library(biomaRt) # 2.44.4
library(pheatmap) # 1.0.12
library(ComplexHeatmap) # 2.9.3
library(circlize) # 0.4.13

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
# do we want to use overlapping genes (between PAXgene and plasma) 
# or keep all genes
# "overlap" or "all_genes"
genes_to_use <- "all_genes"

# all parameter values to test
# minimum number of counts per sample in plasma
min_cnts_per_sample_vals <- c(50000)
# percentage of samples with non-zero counts per gene
min_non_zero_counts_per_genes_vals <- c(20)
# FDR threshold to call a gene differentially expressed
fdr_thresh_vals <- c(0.1)
# TRUE to include Age and Gender
age_sex_model_vals <- c(TRUE)
# target variable
targets_vals <- c("sepsis", "virus")
# options
options_ <- c("a", "b")

# generate a matrix with all possible combinations of the above parameters
comb_mat <- expand.grid(
  "min_cnts_per_sample" = min_cnts_per_sample_vals,
  "min_non_zero_counts_per_genes" = min_non_zero_counts_per_genes_vals,
  "fdr_thresh" = fdr_thresh_vals,
  "age_sex_model" = age_sex_model_vals,
  "target_val" = targets_vals,
  "option_" = options_
)

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

# add virus status
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
write.csv(meta_data, paste(extra_data_path, "processed/EARLI_metadata_adjudication_IDseq_LPSstudyData_7.5.20_viruspos.csv", sep = ""), row.names = FALSE)

# identify genes in common between the plasma and paxgene datasets
genes_in_common <- intersect(rownames(plasma_data), rownames(paxgene_data))

# if we only want to use genes in common between the plasma and PAXgene datasets
if (genes_to_use == "overlap") {
  # "" is for the total counts on the full original dataset
  plasma_data <- plasma_data[rownames(plasma_data) %in% c("", genes_in_common), ]
  paxgene_data <- paxgene_data[rownames(paxgene_data) %in% genes_in_common, ]
}

# filter count data to only keep samples in both the plasma and PAXgene datasets
samples_in_common <- intersect(colnames(plasma_data), colnames(paxgene_data))
plasma_data <- plasma_data[, colnames(plasma_data) %in% samples_in_common]
paxgene_data <- paxgene_data[, colnames(paxgene_data) %in% samples_in_common]

# all ENSG genes
all_ensg_genes <- c(union(rownames(plasma_data), rownames(paxgene_data)))

# extract gene symbols for plasma and paxgene data
# version 103
ensembl <- useEnsembl(
  biomart = "ensembl", dataset = "hsapiens_gene_ensembl",
  version = 103
)
ensembl_res <- getBM(
  values = all_ensg_genes,
  filters = "ensembl_gene_id",
  attributes = c("ensembl_gene_id", "hgnc_symbol", "gene_biotype"),
  mart = ensembl
)
ensembl_res <- ensembl_res[!duplicated(ensembl_res$ensembl_gene_id), ]
rownames(ensembl_res) <- ensembl_res$ensembl_gene_id

# add gene symbols as columns
plasma_data$hgnc_symbol <- ensembl_res[rownames(plasma_data), "hgnc_symbol"]
paxgene_data$hgnc_symbol <- ensembl_res[rownames(paxgene_data), "hgnc_symbol"]

# save outputs to specific folders
results_path <- paste(results_path, paste(genes_to_use, "/", sep=""), sep="")
data_path <- paste(data_path, paste(genes_to_use, "/", sep=""), sep="")

# save full plasma data
# start from row 2 as row 1 is for the total counts per sample
write.csv(plasma_data[2:nrow(plasma_data), ], paste(data_path, paste("processed/plasma_cnts.csv", sep = ""), sep = ""))

# save full paxgene data
write.csv(paxgene_data, paste(data_path, paste("processed/paxgene_cnts.csv", sep = ""), sep = ""))

# plasma G4 samples with 50,000 filter only
# filter plasma samples with less than N protein coding genes-associated counts
sample_names <- colnames(plasma_data[2:ncol(plasma_data)])
samples_to_drop <- sample_names[(plasma_data[1, 2:ncol(plasma_data)] <= 50000)]

plasma_data_filt <- plasma_data[, !(colnames(plasma_data) %in% samples_to_drop)]

# keep only the plasma samples of interest depending on the comparison we want to make
selected_samples <- meta_data[meta_data$sepsis_cat %in% c("4_NO_Sepsis"), "EARLI_Barcode"]
plasma_data_filt <- plasma_data_filt[, colnames(plasma_data_filt) %in% c("hgnc_symbol", selected_samples)]

write.csv(plasma_data_filt, paste(data_path, paste("../processed/50000_plasma_G4_unfiltered_cnts.csv", sep = ""), sep = ""))

# plasma G3 and G5 samples with 50,000 filter only (on full set)
# filter plasma samples with less than N protein coding genes-associated counts
plasma_data_filt <- plasma_data[, !(colnames(plasma_data) %in% samples_to_drop)]

# keep only the plasma samples of interest depending on the comparison we want to make
selected_samples <- meta_data[meta_data$sepsis_cat %in% c("3_Sepsis+Cx-"), "EARLI_Barcode"]
plasma_data_filt <- plasma_data_filt[, colnames(plasma_data_filt) %in% c("hgnc_symbol", selected_samples)]

write.csv(plasma_data_filt, paste(data_path, paste("../processed/50000_plasma_G3_unfiltered_cnts.csv", sep = ""), sep = ""))

# filter plasma samples with less than N protein coding genes-associated counts
plasma_data_filt <- plasma_data[, !(colnames(plasma_data) %in% samples_to_drop)]

# keep only the plasma samples of interest depending on the comparison we want to make
selected_samples <- meta_data[meta_data$sepsis_cat %in% c("5_Unclear"), "EARLI_Barcode"]
plasma_data_filt <- plasma_data_filt[, colnames(plasma_data_filt) %in% c("hgnc_symbol", selected_samples)]

write.csv(plasma_data_filt, paste(data_path, paste("../processed/50000_plasma_G5_unfiltered_cnts.csv", sep = ""), sep = ""))

# for each parameters combination
for (row_ in rownames(comb_mat)) {
  # extract parameter values
  min_cnts_per_sample <- comb_mat[row_, "min_cnts_per_sample"]
  min_non_zero_counts_per_genes <- comb_mat[row_, "min_non_zero_counts_per_genes"]
  fdr_thresh <- comb_mat[row_, "fdr_thresh"]
  age_sex_model <- comb_mat[row_, "age_sex_model"]
  target_val <- comb_mat[row_, "target_val"]
  option_ <- comb_mat[row_, "option_"]
  
  # set two options for sepsis/virus classifiers
  if (option_ == "a"){
    if (target_val=="sepsis"){
      comp_ <- "1vs4"
    }else{
      if (target_val=="virus"){
        comp_ <- "12"
      }
    }
  }else{
    if (option_ == "b"){
      if (target_val=="sepsis"){
        comp_ <- "12vs4"
      }else{
        if (target_val=="virus"){
          comp_ <- "124"
        }
      }
    }
  }

  # generate a unique prefix for that combination
  results_prefix <- paste(as.character(as.integer(min_cnts_per_sample)),
                          paste(paste(min_non_zero_counts_per_genes, fdr_thresh, sep = "_"), 
                                paste(age_sex_model, paste(comp_, target_val, sep="_"), sep = "_"), sep = "_"),
                          sep = "_")

  #########################
  # DATA PREPROCESSING
  #########################
  # filter plasma samples with less than N protein coding genes-associated counts
  sample_names <- colnames(plasma_data[2:ncol(plasma_data)])
  samples_to_drop <- sample_names[(plasma_data[1, 2:ncol(plasma_data)] <= min_cnts_per_sample)]

  plasma_data_filt <- plasma_data[, !(colnames(plasma_data) %in% samples_to_drop)]

  # keep only the plasma samples of interest depending on the comparison we want to make
  if (comp_ == "1vs4") {
    selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "4_NO_Sepsis"), "EARLI_Barcode"]
  } else {
    if ((comp_ == "12vs4") | (comp_ == "124")) {
      selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+", "4_NO_Sepsis"), "EARLI_Barcode"]
    }else{
      if (comp_ == "12"){
        selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+"), "EARLI_Barcode"]
      }
    }
  }
  plasma_data_filt <- plasma_data_filt[, colnames(plasma_data_filt) %in% c("hgnc_symbol", selected_samples)]
  plasma_data_low <- plasma_data[, colnames(plasma_data) %in% c("hgnc_symbol", intersect(selected_samples, samples_to_drop))]

  # filter genes present in only some of the plasma samples (use only samples with more than the minimum count)
  # ignore first row (gene counts per sample) and last column (gene symbols)
  gene_names <- rownames(plasma_data_filt[2:nrow(plasma_data_filt), 1:(ncol(plasma_data_filt) - 1)])
  gene_data <- plasma_data_filt[2:nrow(plasma_data_filt), 1:(ncol(plasma_data_filt) - 1)]
  genes_to_drop <- gene_names[apply(gene_data, 1, function(x) sum(x > 0)) / (ncol(plasma_data_filt) - 1) < (min_non_zero_counts_per_genes / 100)]
  plasma_data_filt <- plasma_data_filt[!(rownames(plasma_data_filt) %in% genes_to_drop), ]
  plasma_data_low <- plasma_data_low[!(rownames(plasma_data_low) %in% genes_to_drop), ]

  # filter the meta data to match the selected samples
  # make sure our variable of interest is a factor
  meta_data_filt <- meta_data[meta_data$EARLI_Barcode %in% colnames(plasma_data_filt), ]
  meta_data_filt$sepsis_cat <- as.factor(meta_data_filt$sepsis_cat)
  meta_data_filt$viruspos <- as.factor(meta_data_filt$viruspos)

  meta_data_low <- meta_data[meta_data$EARLI_Barcode %in% colnames(plasma_data_low), ]
  meta_data_low$sepsis_cat <- as.factor(meta_data_low$sepsis_cat)
  meta_data_low$viruspos <- as.factor(meta_data_low$viruspos)

  # for paxgene data, filter samples to keep the same ones as the ones selected from the plasma data
  paxgene_data_filt <- paxgene_data[, colnames(paxgene_data) %in% c(meta_data_filt$EARLI_Barcode, "hgnc_symbol")]
  paxgene_data_low <- paxgene_data[, colnames(paxgene_data) %in% c(meta_data_low$EARLI_Barcode, "hgnc_symbol")]

  # if we only consider overlapping genes
  if (genes_to_use == "overlap") {
    paxgene_data_filt <- paxgene_data_filt[rownames(paxgene_data_filt) %in% rownames(plasma_data_filt), ]
    paxgene_data_low <- paxgene_data_low[rownames(paxgene_data_low) %in% rownames(plasma_data_low), ]
  }

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
    meta_data_low$Age[is.na(meta_data_low$Age) & meta_data_low[, target_col] == target_val_cat] <-
      mean(meta_data_low$Age[(!is.na(meta_data_low$Age)) & meta_data_low[, target_col] == target_val_cat])
  }

  # standard scaling on Age
  meta_data_filt$age_scaled <- scale(meta_data_filt$Age)
  meta_data_low$age_scaled <- scale(meta_data_low$Age)

  # make sure gender is a factor
  meta_data_filt$gender <- as.factor(meta_data_filt$Gender)
  meta_data_low$gender <- as.factor(meta_data_low$Gender)

  # drop first row and column and store gene symbols
  plasma_data_filt <- plasma_data_filt[2:nrow(plasma_data_filt), ]
  plasma_data_filt_symbols <- plasma_data_filt
  gene_symbols_plasma <- plasma_data_filt$hgnc_symbol
  plasma_data_filt <- plasma_data_filt[, 1:(ncol(plasma_data_filt) - 1)]

  plasma_data_low <- plasma_data_low[2:nrow(plasma_data_low), ]
  plasma_data_low_symbols <- plasma_data_low
  plasma_data_low <- plasma_data_low[, 1:(ncol(plasma_data_low) - 1)]

  paxgene_data_filt_symbols <- paxgene_data_filt
  gene_symbols_paxgene <- paxgene_data_filt$hgnc_symbol
  paxgene_data_filt <- paxgene_data_filt[, 1:(ncol(paxgene_data_filt) - 1)]

  paxgene_data_low_symbols <- paxgene_data_low
  paxgene_data_low <- paxgene_data_low[, 1:(ncol(paxgene_data_low) - 1)]

  # save the counts data with gene symbols
  write.csv(plasma_data_filt_symbols, paste(data_path, paste("processed/", paste(results_prefix, "plasma_cnts.csv", sep = "_"), sep = ""), sep = ""))
  write.csv(plasma_data_low_symbols, paste(data_path, paste("processed/", paste(results_prefix, "plasma_cnts_low.csv", sep = "_"), sep = ""), sep = ""))

  write.csv(paxgene_data_filt_symbols, paste(data_path, paste("processed/", paste(results_prefix, "paxgene_cnts.csv", sep = "_"), sep = ""), sep = ""))
  write.csv(paxgene_data_low_symbols, paste(data_path, paste("processed/", paste(results_prefix, "paxgene_cnts_low.csv", sep = "_"), sep = ""), sep = ""))

  write.csv(meta_data_low, paste(data_path, paste("processed/", paste(results_prefix, "metadata_low.csv", sep = "_"), sep = ""), sep = ""))

  #########################
  # DE ANALYSIS
  #########################
  # build DESeq2 object
  meta_data_tmp <- meta_data_filt
  
  # for the classifiers, we only care about sepsis vs non-sepsis
  meta_data_tmp$sepsis_cat[meta_data_tmp$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+")] <- "1_Sepsis+BldCx+"
  
  # save metadata
  write.csv(meta_data_tmp, paste(data_path, paste("processed/", paste(results_prefix, "metadata.csv", sep = "_"), sep = ""), sep = ""))
  
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
  if (age_sex_model) {
    dds_plasma <- DESeqDataSetFromMatrix(
      countData = plasma_data_filt,
      colData = meta_data_tmp,
      design = ~ target_val + age_scaled + gender
    )

    dds_paxgene <- DESeqDataSetFromMatrix(
      countData = paxgene_data_filt,
      colData = meta_data_tmp,
      design = ~ target_val + age_scaled + gender
    )
  } else {
    dds_plasma <- DESeqDataSetFromMatrix(
      countData = plasma_data_filt,
      colData = meta_data_tmp,
      design = ~ target_val
    )

    dds_paxgene <- DESeqDataSetFromMatrix(
      countData = paxgene_data_filt,
      colData = meta_data_tmp,
      design = ~ target_val
    )
  }

  # choose the reference level for the factor of interest
  if (target_val == "sepsis"){
    dds_plasma$target_val <- relevel(dds_plasma$target_val, ref = "4_NO_Sepsis")
    dds_paxgene$target_val <- relevel(dds_paxgene$target_val, ref = "4_NO_Sepsis")
  }else{
    if (target_val == "virus"){
      dds_plasma$target_val <- relevel(dds_plasma$target_val, ref = "nonviral")
      dds_paxgene$target_val <- relevel(dds_paxgene$target_val, ref = "nonviral")
    }
  }

  # run DESeq
  dds_plasma <- DESeq(dds_plasma)
  dds_paxgene <- DESeq(dds_paxgene)

  # transform data for PCA plot and for post-classification visualisation
  vsd_plasma <- vst(dds_plasma, blind = TRUE)
  vsd_paxgene <- vst(dds_paxgene, blind = TRUE)

  # save VST data
  write.csv(assay(vsd_plasma), paste(data_path, paste("processed/", paste(results_prefix, "plasma_vsd.csv", sep = "_"), sep = ""), sep = ""))
  write.csv(assay(vsd_paxgene), paste(data_path, paste("processed/", paste(results_prefix, "paxgene_vsd.csv", sep = "_"), sep = ""), sep = ""))
  
  vsd_plasma_tmp <- as.data.frame(assay(vsd_plasma))
  vsd_plasma_tmp$hgnc_symbol <- gene_symbols_plasma
  write.csv(vsd_plasma_tmp, paste(data_path, paste("processed/", paste(results_prefix, "plasma_vsd_with_hgnc_symbols.csv", sep = "_"), sep = ""), sep = ""))

  # plot a PCA with group labels overlaid
  # for the plasma data
  pca_data_plasma <- t(assay(vsd_plasma))
  pca_data_plasma <- pca_data_plasma[, colnames(pca_data_plasma)[apply(pca_data_plasma, 2, function(x) length(unique(x))) > 1]]
  pca_res <- prcomp(pca_data_plasma, scale. = TRUE, center = TRUE)
  pdf(paste(results_path, paste(results_prefix, "pca_plasma_vst.pdf", sep = "_"), sep = ""),
    width = 6, height = 5
  )
  print(autoplot(pca_res,
    data =
      meta_data_tmp, colour = "target_val"
  ))
  dev.off()

  # for the paxgene data
  pca_data_paxgene <- t(assay(vsd_paxgene))
  pca_data_paxgene <- pca_data_paxgene[, colnames(pca_data_paxgene)[apply(pca_data_paxgene, 2, function(x) length(unique(x))) > 1]]
  pca_res <- prcomp(pca_data_paxgene, scale. = TRUE, center = TRUE)
  pdf(paste(results_path, paste(results_prefix, "pca_paxgene_vst.pdf", sep = "_"), sep = ""),
    width = 6, height = 5
  )
  print(autoplot(pca_res,
    data =
      meta_data_tmp, colour = "target_val"
  ))
  dev.off()

  # extract DESeq2 results
  res_plasma <- lfcShrink(dds_plasma,coef=2,type ="apeglm") 
  if (target_val == "sepsis"){
    #res_plasma <- results(dds_plasma, contrast = c("target_val", "1_Sepsis+BldCx+", "4_NO_Sepsis"))
    res_paxgene <- results(dds_paxgene, contrast = c("target_val", "1_Sepsis+BldCx+", "4_NO_Sepsis"))
  }else{
    if (target_val == "virus"){
      #res_plasma <- results(dds_plasma, contrast = c("target_val", "viral", "nonviral"))
      res_paxgene <- results(dds_paxgene, contrast = c("target_val", "viral", "nonviral"))
    }
  }

  # add gene symbols and biotype
  res_plasma$hgnc_symbol <- gene_symbols_plasma
  res_paxgene$hgnc_symbol <- gene_symbols_paxgene
  
  res_plasma$gene_biotype <- ensembl_res[rownames(res_plasma), "gene_biotype"]
  res_paxgene$gene_biotype <- ensembl_res[rownames(res_paxgene), "gene_biotype"]

  # sort the genes from lowest to highest given adjusted p-values
  res_plasma <- res_plasma[order(res_plasma$padj, decreasing = F), ]
  res_paxgene <- res_paxgene[order(res_paxgene$padj, decreasing = F), ]

  # replace NA values with 1s and keep only significant genes
  res_plasma$padj[is.na(res_plasma$padj)] <- 1
  sig_results_plasma <- data.frame(res_plasma[res_plasma$padj < fdr_thresh, ])

  res_paxgene$padj[is.na(res_paxgene$padj)] <- 1
  sig_results_paxgene <- data.frame(res_paxgene[res_paxgene$padj < fdr_thresh, ])

  # save the output as a CSV file
  write.csv(sig_results_plasma, paste(results_path, paste(results_prefix, "plasma_DGEA_results.csv", sep = "_"), sep = ""))
  write.csv(sig_results_paxgene, paste(results_path, paste(results_prefix, "paxgene_DGEA_results.csv", sep = "_"), sep = ""))

  # save unfiltered outputs (for IPA)
  write.csv(data.frame(res_plasma), paste(results_path, paste(results_prefix, "plasma_DGEA_unfiltered_results.csv", sep = "_"), sep = ""))
  write.csv(data.frame(res_paxgene), paste(results_path, paste(results_prefix, "paxgene_DGEA_unfiltered_results.csv", sep = "_"), sep = ""))
  
  # generate a volcano plot
  # only display top 25 gene symbols
  # for plasma data
  res_plasma_df <- data.frame(res_plasma)
  res_plasma_df$sig <- res_plasma_df$padj < fdr_thresh
  pdf(paste(results_path, paste(results_prefix, "plasma_volcano_plot.pdf", sep = "_"), sep = ""), width = 6, height = 6)
  p <- ggplot(res_plasma_df, aes(log2FoldChange, -log10(pvalue))) +
    geom_point(aes(col = sig)) +
    scale_color_manual(values = c("black", "red")) +
    ggtitle("Volcano Plot") +
    theme(legend.position = "none") +
    geom_text_repel(
      data = res_plasma_df[1:25, ],
      aes(label = res_plasma_df[1:25, "hgnc_symbol"]),
      max.overlaps = 25
    )
  print(p)
  dev.off()

  # for paxgene data
  res_paxgene_df <- data.frame(res_paxgene)
  res_paxgene_df$sig <- res_paxgene_df$padj < fdr_thresh
  pdf(paste(results_path, paste(results_prefix, "paxgene_volcano_plot.pdf", sep = "_"), sep = ""), width = 6, height = 6)
  p <- ggplot(res_paxgene_df, aes(log2FoldChange, -log10(pvalue))) +
    geom_point(aes(col = sig)) +
    scale_color_manual(values = c("black", "red")) +
    ggtitle("Volcano Plot") +
    theme(legend.position = "none") +
    geom_text_repel(
      data = res_paxgene_df[1:25, ],
      aes(label = res_paxgene_df[1:25, "hgnc_symbol"]),
      max.overlaps = 25
    )
  print(p)
  dev.off()
  
  # generate heatmaps
  annot_data <- meta_data_tmp[, target_col, drop=F]
  annot_data[, target_col] <- as.factor(annot_data[, target_col])
  
  if (target_val=="sepsis"){
    annot_data[, target_col] <- relevel(annot_data[, target_col], ref = "4_NO_Sepsis")
  }else{
    if (target_val=="virus"){
      annot_data[, target_col] <- relevel(annot_data[, target_col], ref = "nonviral")
    }
  }
  
  # for plasma
  hm_data_plasma <- assay(vsd_plasma)
  hm_data_plasma <- hm_data_plasma[rownames(res_plasma_df[1:50, ]), ]
  
  # add hgnc symbols
  rownames(hm_data_plasma) <- 
    sapply(rownames(hm_data_plasma), function(x){
      hgnc_symbol_tmp <- res_plasma_df[x, "hgnc_symbol"]
      if (hgnc_symbol_tmp==""){
        return(x)
      }else{
        return(hgnc_symbol_tmp)
      }
    })
  
  # order columns by annot_data
  hm_data_plasma <- hm_data_plasma[, order(annot_data[, target_col])]
  annot_data <- annot_data[colnames(hm_data_plasma), , drop=F]
  
  if (target_val ==  "sepsis"){
    levels(annot_data[, target_col]) <- c(levels(annot_data[, target_col]), "1_Sepsis+BldCx+ \n2_Sepsis+OtherCx+")
    annot_data[, target_col][annot_data[, target_col]=="1_Sepsis+BldCx+"] <- "1_Sepsis+BldCx+ \n2_Sepsis+OtherCx+" 
    annot_data[, target_col] <- droplevels(annot_data[, target_col])
    
    col_vect <- c("#06788f", "#8b02bd")
    names(col_vect) <- levels(annot_data[, target_col])
    col_vect <- list("sepsis_cat"=col_vect)
    
    column_ha <- HeatmapAnnotation(sepsis_cat = annot_data[, target_col],
                                   annotation_legend_param = list(nrow=2),
                                   col = col_vect)
  }else{
    if (target_val=="virus"){
      col_vect <- c("#06788f", "#8b02bd")
      names(col_vect) <- levels(annot_data[, target_col])
      col_vect <- list("viruspos"=col_vect)
      
      column_ha <- HeatmapAnnotation(viruspos = annot_data[, target_col],
                                     annotation_legend_param = list(nrow=2),
                                     col = col_vect)
    }
  }
  
  # row scaling
  hm_data_plasma <- t(scale(t(hm_data_plasma)))
  
  # legend
  col_fun <- colorRamp2(c(min(hm_data_plasma), 0, max(hm_data_plasma)), c("blue", "white", "red"))
  lgd <- Legend(col_fun = col_fun, title = "expression")
  
  set.seed(123)
  hm <- Heatmap(hm_data_plasma, name="expression", 
                top_annotation = column_ha, column_split=annot_data[, target_col],
                column_title = " ",
                cluster_columns= FALSE, show_column_names = FALSE)
  
  pdf(paste(results_path, paste(results_prefix, "plasma_hm.pdf", sep = "_"), sep = ""),
      width = 10, height = 8)
  draw(hm, merge_legend = TRUE)
  dev.off()
  
  # for paxgene
  hm_data_paxgene <- assay(vsd_paxgene)
  hm_data_paxgene <- hm_data_paxgene[rownames(res_paxgene_df[1:50, ]), ]
  
  # add hgnc symbols
  rownames(hm_data_paxgene) <- 
    sapply(rownames(hm_data_paxgene), function(x){
      hgnc_symbol_tmp <- res_paxgene_df[x, "hgnc_symbol"]
      if (hgnc_symbol_tmp==""){
        return(x)
      }else{
        return(hgnc_symbol_tmp)
      }
    })
  
  # order columns by annot_data
  hm_data_paxgene <- hm_data_paxgene[, order(annot_data[, target_col])]
  annot_data <- annot_data[colnames(hm_data_paxgene), , drop=F]

  if (target_val ==  "sepsis"){
    levels(annot_data[, target_col]) <- c(levels(annot_data[, target_col]), "1_Sepsis+BldCx+ \n2_Sepsis+OtherCx+")
    annot_data[, target_col][annot_data[, target_col]=="1_Sepsis+BldCx+"] <- "1_Sepsis+BldCx+ \n2_Sepsis+OtherCx+" 
    annot_data[, target_col] <- droplevels(annot_data[, target_col])
    
    col_vect <- c("#06788f", "#8b02bd")
    names(col_vect) <- levels(annot_data[, target_col])
    col_vect <- list("sepsis_cat"=col_vect)
    
    column_ha <- HeatmapAnnotation(sepsis_cat = annot_data[, target_col],
                                   annotation_legend_param = list(nrow=2),
                                   col = col_vect)
  }else{
    if (target_val=="virus"){
      col_vect <- c("#06788f", "#8b02bd")
      names(col_vect) <- levels(annot_data[, target_col])
      col_vect <- list("viruspos"=col_vect)
      
      column_ha <- HeatmapAnnotation(viruspos = annot_data[, target_col],
                                     annotation_legend_param = list(nrow=2),
                                     col = col_vect)
    }
  }
  
  # row scaling
  hm_data_paxgene <- t(scale(t(hm_data_paxgene)))
  
  # legend
  col_fun <- colorRamp2(c(min(hm_data_paxgene), 0, max(hm_data_paxgene)), c("blue", "white", "red"))
  lgd <- Legend(col_fun = col_fun, title = "expression")
  
  set.seed(123)
  hm <- Heatmap(hm_data_paxgene, name="expression", 
                top_annotation = column_ha, column_split=annot_data[, target_col],
                column_title = " ",
                cluster_columns= FALSE, show_column_names = FALSE)
  
  pdf(paste(results_path, paste(results_prefix, "paxgene_hm.pdf", sep = "_"), sep = ""),
      width = 10, height = 8)
  draw(hm, merge_legend = TRUE)
  dev.off()
}
