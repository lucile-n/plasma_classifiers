######################################################
# 1.de_script.R
# created on March 15 2021
# lucile.neyton@ucsf.edu

# This script aims at performing a differential
#   expression analysis between samples,
#   given their sepsis status (1=sepsis+bacteremia, 4=no sepsis)

# Input files (data folder):
# - CSV-formatted file containing gene counts
#   (genes x samples)
# - CSV-formatted files containing sepsis status
#   and Sex and Age (samples x label)

# Output files (results folder):
# - One file with counts values per run.
# - One file with normnalised (unscaled) gene expression values per run.
# - One metadata file per run.
# - One CSV file with significantly differentially expressed genes per run.
######################################################

rm(list = ls())
setwd("/Users/lucileneyton/OneDrive\ -\ University\ of\ California,\ San\ Francisco/UCSF/EARLI_plasma/")

# load libraries
library(DESeq2) # 1.28.1
library(readxl) # 1.3.1
library(ggfortify) # 0.4.11

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
min_cnts_per_sample_vals <- c(50000, 100000)
min_non_zero_counts_per_genes_vals <- c(20, 30, 40, 50)
fdr_thresh_vals <- c(0.01, 0.05, 0.1)
age_sex_model_vals <- c(TRUE, FALSE)

comb_mat <- expand.grid("min_cnts_per_sample" = min_cnts_per_sample_vals, 
                        "min_non_zero_counts_per_genes" = min_non_zero_counts_per_genes_vals, 
                        "fdr_thresh" = fdr_thresh_vals,
                        "age_sex_model" = age_sex_model_vals)

#########################
# DATA LOADING
#########################
# list data files
cnt_data_path <- paste(data_path, "raw/earli_plasma_counts.csv", sep = "")
# Sepsis status
sepsis_status_data_path <- paste(data_path, "raw/EARLI_metadata_plasma_only.xlsx", sep = "")
# Age and sex status
age_sex_data_path <- paste(extra_data_path, "raw/EARLI_sampletracking_phenotyping/FinalData_04172020.xlsx", sep = "")

# read data in
cnt_data <- read.csv(cnt_data_path, row.names = 1)

sepsis_status_data <- read_excel(sepsis_status_data_path, sheet=1)
age_sex_data <- read_excel(age_sex_data_path, sheet=1)

for (row_ in rownames(comb_mat)){
  min_cnts_per_sample <- comb_mat[row_, "min_cnts_per_sample"]
  min_non_zero_counts_per_genes <- comb_mat[row_, "min_non_zero_counts_per_genes"]
  fdr_thresh <- comb_mat[row_, "fdr_thresh"]
  age_sex_model <- comb_mat[row_, "age_sex_model"]
  
  results_prefix <- paste(as.character(as.integer(min_cnts_per_sample)), 
                          paste(paste(min_non_zero_counts_per_genes, fdr_thresh, sep="_"), age_sex_model, sep="_"), 
                          sep="_")
  
  #########################
  # DATA PREPROCESSING
  #########################
  # add a column name for sepsis category
  colnames(sepsis_status_data)[4] <- "sepsis_cat"
  
  # merge sepsis status and age/sex data files
  meta_data <- merge(sepsis_status_data, age_sex_data, by.x="PatientID", by.y="barcode1", all=T)
  
  # filter samples with less than 100,000 protein coding genes-associated counts
  sample_names <- colnames(cnt_data[2:236])
  samples_to_drop <- sample_names[(cnt_data[1,2:236]<=min_cnts_per_sample)]
  
  cnt_data_filt <- cnt_data[,!(colnames(cnt_data) %in% samples_to_drop)]
  
  # keep only the samples of interest
  selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "4_NO_Sepsis"), "SampleID"]
  
  cnt_data_filt <- cnt_data_filt[,colnames(cnt_data_filt) %in% c("gene_symbol", selected_samples)]
  
  # filter genes present in only some of the samples
  gene_names <- rownames(cnt_data_filt[2:nrow(cnt_data_filt),2:ncol(cnt_data_filt)])
  gene_data <- cnt_data_filt[2:nrow(cnt_data_filt),2:ncol(cnt_data_filt)]
  genes_to_drop <- gene_names[apply(gene_data, 1, function(x) sum(x>0))/(ncol(cnt_data_filt)-1)<(min_non_zero_counts_per_genes/100)]
  
  cnt_data_filt <- cnt_data_filt[!(rownames(cnt_data_filt) %in% genes_to_drop), ]
    
  # filter metadata make sure our variable of interest is a factor
  meta_data_filt <- meta_data[meta_data$SampleID %in% colnames(cnt_data_filt),]
  meta_data_filt$sepsis_cat <- as.factor(meta_data_filt$sepsis_cat)

  # replace missing age values by average for that subgroup
  meta_data_filt$age[is.na(meta_data_filt$age) & meta_data_filt$sepsis_cat=="1_Sepsis+BldCx+"] <- 
    mean(meta_data_filt$age[(!is.na(meta_data_filt$age)) & meta_data_filt$sepsis_cat=="1_Sepsis+BldCx+"])
  meta_data_filt$age[is.na(meta_data_filt$age) & meta_data_filt$sepsis_cat=="4_NO_Sepsis"] <- 
    mean(meta_data_filt$age[(!is.na(meta_data_filt$age)) & meta_data_filt$sepsis_cat=="4_NO_Sepsis"])
  
  # standard scaling on Age
  meta_data_filt$age_scaled <- scale(meta_data_filt$age)
  
  # make sure gender is a factor
  meta_data_filt$gender <- as.factor(meta_data_filt$gender)
  
  # drop first row and column and store gene symbols
  cnt_data_filt <- cnt_data_filt[2:nrow(cnt_data_filt),]
  gene_symbols <- cnt_data_filt$gene_symbol
  cnt_data_filt <- cnt_data_filt[,2:ncol(cnt_data_filt)]
  
  cnt_data_filt_symbols <- cnt_data_filt
  cnt_data_filt_symbols$hgnc_symbol <- gene_symbols
  write.csv(cnt_data_filt_symbols, paste(data_path, paste("processed/", paste(results_prefix, "cnts.csv", sep="_"), sep=""), sep = ""))
  
  #########################
  # DE ANALYSIS
  #########################
  # build DESeq2 object
  if (age_sex_model){
    dds <- DESeqDataSetFromMatrix(countData = cnt_data_filt,
                                  colData = meta_data_filt,
                                  design = ~ sepsis_cat + age_scaled + gender)
  }else{
    dds <- DESeqDataSetFromMatrix(countData = cnt_data_filt,
                                  colData = meta_data_filt,
                                  design = ~ sepsis_cat)
  }
  
  
  # choose the reference level for the factor of interest
  dds$sepsis_cat <- relevel(dds$sepsis_cat, ref = "4_NO_Sepsis")
  
  # run DESeq
  dds <- DESeq(dds)
  
  # transform data for PCA plot
  vsd <- vst(dds, blind = TRUE)
  
  # save metadata
  write.csv(assay(vsd), paste(data_path, paste("processed/", paste(results_prefix, "vsd.csv", sep="_"), sep=""), sep = ""))
  
  # plot a PCA with LCA labels overlaid
  pca_res <- prcomp(t(assay(vsd)), scale. = TRUE, center = TRUE)
  pdf(paste(results_path, paste(results_prefix, "pca_vst.pdf", sep="_"), sep = ""),
      width = 6, height = 5
  )
  print(autoplot(pca_res, data =
                   meta_data_filt, colour = "sepsis_cat"))
  dev.off()
  
  # save metadata
  write.csv(meta_data_filt, paste(data_path, paste("processed/", paste(results_prefix, "metadata_1vs4.csv", sep="_"), sep=""), sep = ""))
  
  # extract results
  res <- results(dds, contrast = c("sepsis_cat", "1_Sepsis+BldCx+", "4_NO_Sepsis"))
  
  # add gene symbols
  res$hgnc_symbol <- gene_symbols
  
  # sort the genes from lowest to highest given adjusted p-values
  res <- res[order(res$padj, decreasing = F), ]
  
  # replace NA values with 1s
  res$padj[is.na(res$padj)] <- 1
  sig_results <- data.frame(res[res$padj < fdr_thresh, ])
  
  # save the output as a CSV file
  write.csv(sig_results, paste(results_path, paste(results_prefix, "DGEA_results.csv", sep="_"), sep = ""))
  
  # generate a volcano plot
  # only display top 25 gene symbols
  res_df <- data.frame(res)
  res_df$sig <- res_df$padj < fdr_thresh
  pdf(paste(results_path, paste(results_prefix, "volcano_plot.pdf", sep="_"), sep = ""), width = 5, height = 5)
  p <- ggplot(res_df, aes(log2FoldChange, -log10(pvalue))) +
    geom_point(aes(col = sig)) +
    scale_color_manual(values = c("black", "red")) +
    ggtitle("Volcano Plot") +
    theme(legend.position = "none") +
    geom_text_repel(data = res_df[1:25, ],
                    aes(label = res_df[1:25, "hgnc_symbol"]))
  print(p)
  dev.off()
} 
  
