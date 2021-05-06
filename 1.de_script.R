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
min_cnts_per_sample_vals <- c(50000)
min_non_zero_counts_per_genes_vals <- c(20, 30, 40, 50)
fdr_thresh_vals <- c(0.1)
age_sex_model_vals <- c(TRUE, FALSE)
groups_to_compare_vals <- c("1vs4", "12vs4")

comb_mat <- expand.grid("min_cnts_per_sample" = min_cnts_per_sample_vals, 
                        "min_non_zero_counts_per_genes" = min_non_zero_counts_per_genes_vals, 
                        "fdr_thresh" = fdr_thresh_vals,
                        "age_sex_model" = age_sex_model_vals,
                        "groups_to_compare" = groups_to_compare_vals)

#########################
# DATA LOADING
#########################
# list data files
cnt_data_path <- paste(data_path, "raw/earli_plasma_counts.csv", sep = "")
# Sepsis status
sepsis_status_data_path <- paste(data_path, "raw/EARLI_metadata_plasma_only.xlsx", sep = "")
# Age and sex status
age_sex_data_path <- paste(extra_data_path, "raw/EARLI_sampletracking_phenotyping/FinalData_04172020.xlsx", sep = "")

# paxgene data
# select only overlapping samples
paxgene_data_path <- paste(data_path, "raw/EARLI_star_pc_and_lincRNA_genecounts.qc.tsv", sep = "")
# meta data for paxgene data
paxgene_metadata_path <- paste(extra_data_path, "processed/EARLI_metadata_adjudication_IDseq_LPSstudyData_7.5.20.csv", sep="")

# read data in
cnt_data <- read.csv(cnt_data_path, row.names = 1)

sepsis_status_data <- read_excel(sepsis_status_data_path, sheet=1)
age_sex_data <- read_excel(age_sex_data_path, sheet=1)

# add a column name for sepsis category
colnames(sepsis_status_data)[4] <- "sepsis_cat"

# merge sepsis status and age/sex data files
meta_data <- merge(sepsis_status_data, age_sex_data, by.x="PatientID", by.y="barcode1", all=T)

# read-in paxgene data to keep only samples in common
paxgene_data <- read.table(paxgene_data_path, row.names = 1, sep="\t")
paxgene_metadata <- read.csv(paxgene_metadata_path)
paxgene_metadata$EARLI_Barcode <- sapply(paxgene_metadata$Barcode, function(x) paste("EARLI", x, sep="_"))

# keep only genes in common between the plasma and PAXgene samples
tmp_paxgene_rows <- sapply(rownames(paxgene_data), function(x) strsplit(x, ".", fixed = TRUE)[[1]][1])
genes_in_common <- intersect(rownames(cnt_data), tmp_paxgene_rows)

# "" is for the total counts on the full original dataset
cnt_data <- cnt_data[rownames(cnt_data) %in% c("", genes_in_common), ]
paxgene_data <- paxgene_data[tmp_paxgene_rows %in% genes_in_common, ]

# filter count data to only keep samples with a PAXgene tube
cnt_data_all <- cnt_data
to_keep <- sapply(colnames(paxgene_data), function(x) paste(strsplit(x, "_")[[1]][1], strsplit(x, "_")[[1]][2], sep="_"))
cnt_data <- cnt_data[, colnames(cnt_data) %in% to_keep]

# drop duplicated ENSG identifiers
paxgene_data <- paxgene_data[!grepl(pattern = ".*(\\.).*(\\.).*", rownames(paxgene_data)), ]
rownames(paxgene_data) <- sapply(rownames(paxgene_data), function(x) strsplit(x, "\\.")[[1]][1])

# all ENSG genes
all_ensg_genes <- c(union(rownames(cnt_data), rownames(paxgene_data)))

# extract gene symbols for plasma and paxgene data
ensembl <- useEnsembl(
  biomart = "ensembl", dataset = "hsapiens_gene_ensembl",
  version = 103
) # version 103
ensembl_res <- getBM(
  values = all_ensg_genes,
  filters = "ensembl_gene_id",
  attributes = c("ensembl_gene_id", "hgnc_symbol"), 
  mart = ensembl
)
ensembl_res <- ensembl_res[!duplicated(ensembl_res$ensembl_gene_id), ]
rownames(ensembl_res) <- ensembl_res$ensembl_gene_id

# add gene symbols as columns
cnt_data$hgnc_symbol <- ensembl_res[rownames(cnt_data), "hgnc_symbol"]
paxgene_data$hgnc_symbol <- ensembl_res[rownames(paxgene_data), "hgnc_symbol"]
cnt_data_all$hgnc_symbol <- ensembl_res[rownames(cnt_data_all), "hgnc_symbol"]

# save full plasma data
write.csv(cnt_data_all[2:nrow(cnt_data_all),], paste(data_path, paste("processed/plasma_cnts.csv", sep=""), sep = ""))

# save full paxgene data
write.csv(paxgene_data, paste(data_path, paste("processed/paxgene_cnts.csv", sep=""), sep = ""))

for (row_ in rownames(comb_mat)){
  min_cnts_per_sample <- comb_mat[row_, "min_cnts_per_sample"]
  min_non_zero_counts_per_genes <- comb_mat[row_, "min_non_zero_counts_per_genes"]
  fdr_thresh <- comb_mat[row_, "fdr_thresh"]
  age_sex_model <- comb_mat[row_, "age_sex_model"]
  groups_to_compare <- comb_mat[row_, "groups_to_compare"][[1]]
  
  results_prefix <- paste(as.character(as.integer(min_cnts_per_sample)), 
                          paste(paste(min_non_zero_counts_per_genes, fdr_thresh, sep="_"), paste(age_sex_model, groups_to_compare, sep="_"), sep="_"), 
                          sep="_")
  
  #########################
  # DATA PREPROCESSING
  #########################
  # filter samples with less than N protein coding genes-associated counts
  sample_names <- colnames(cnt_data[2:ncol(cnt_data)])
  samples_to_drop <- sample_names[(cnt_data[1,2:ncol(cnt_data)]<=min_cnts_per_sample)]
  
  cnt_data_filt <- cnt_data[, !(colnames(cnt_data) %in% samples_to_drop)]
  
  # keep only the samples of interest
  if (groups_to_compare == "1vs4"){
    selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "4_NO_Sepsis"), "SampleID"]
  }else{
    if (groups_to_compare == "12vs4"){
      selected_samples <- meta_data[meta_data$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+", "4_NO_Sepsis"), "SampleID"]
    }
  }
  
  cnt_data_filt <- cnt_data_filt[, colnames(cnt_data_filt) %in% c("hgnc_symbol", selected_samples)]
  cnt_data_low <- cnt_data_all[, colnames(cnt_data_all) %in% c("hgnc_symbol", intersect(selected_samples, samples_to_drop))]
  
  # filter genes present in only some of the samples
  gene_names <- rownames(cnt_data_filt[2:nrow(cnt_data_filt), 1:(ncol(cnt_data_filt)-1)])
  gene_data <- cnt_data_filt[2:nrow(cnt_data_filt), 1:(ncol(cnt_data_filt)-1)]
  genes_to_drop <- gene_names[apply(gene_data, 1, function(x) sum(x>0))/(ncol(cnt_data_filt)-1)<(min_non_zero_counts_per_genes/100)]
  
  cnt_data_filt <- cnt_data_filt[!(rownames(cnt_data_filt) %in% genes_to_drop), ]
  cnt_data_low <- cnt_data_low[!(rownames(cnt_data_low) %in% genes_to_drop), ]
    
  # filter metadata make sure our variable of interest is a factor
  meta_data_filt <- meta_data[meta_data$SampleID %in% colnames(cnt_data_filt),]
  meta_data_filt$sepsis_cat <- as.factor(meta_data_filt$sepsis_cat)
  
  meta_data_low <- meta_data[meta_data$SampleID %in% colnames(cnt_data_low),]
  meta_data_low$sepsis_cat <- as.factor(meta_data_low$sepsis_cat)
  
  # for paxgene data, filter genes
  paxgene_data_filt <- paxgene_data

  paxgene_metadata_filt <- paxgene_metadata[(paxgene_metadata$EARLI_Barcode %in% meta_data_filt$SampleID), ]
  paxgene_data_filt <- paxgene_data_filt[, colnames(paxgene_data_filt) %in% c(paxgene_metadata_filt$HOST_PAXgene_filename, "hgnc_symbol")]
  
  colnames(paxgene_data_filt) <- sapply(colnames(paxgene_data_filt), 
                                        function(x) paste(strsplit(x, "_")[[1]][1], strsplit(x, "_")[[1]][2], sep="_"))
  
  paxgene_data_filt <- paxgene_data_filt[rownames(paxgene_data_filt) %in% rownames(cnt_data_filt), ]
  
  # for paxgene data low, filter genes
  paxgene_data_low <- paxgene_data
  
  paxgene_metadata_low <- paxgene_metadata[(paxgene_metadata$EARLI_Barcode %in% meta_data_low$SampleID), ]
  paxgene_data_low <- paxgene_data_low[, colnames(paxgene_data_low) %in% c(paxgene_metadata_low$HOST_PAXgene_filename, "hgnc_symbol")]
  
  colnames(paxgene_data_low) <- sapply(colnames(paxgene_data_low), 
                                        function(x) paste(strsplit(x, "_")[[1]][1], strsplit(x, "_")[[1]][2], sep="_"))
  
  paxgene_data_low <- paxgene_data_low[rownames(paxgene_data_low) %in% rownames(cnt_data_low), ]

  # replace missing age values by average for that subgroup
  for (sepsis_cat in unique(meta_data_filt$sepsis_cat)){
    meta_data_filt$age[is.na(meta_data_filt$age) & meta_data_filt$sepsis_cat==sepsis_cat] <- 
      mean(meta_data_filt$age[(!is.na(meta_data_filt$age)) & meta_data_filt$sepsis_cat==sepsis_cat])
    meta_data_low$age[is.na(meta_data_low$age) & meta_data_low$sepsis_cat==sepsis_cat] <- 
      mean(meta_data_low$age[(!is.na(meta_data_low$age)) & meta_data_low$sepsis_cat==sepsis_cat])
  }
  
  # standard scaling on Age
  meta_data_filt$age_scaled <- scale(meta_data_filt$age)
  meta_data_low$age_scaled <- scale(meta_data_low$age)
  
  # make sure gender is a factor
  meta_data_filt$gender <- as.factor(meta_data_filt$gender)
  meta_data_low$gender <- as.factor(meta_data_low$gender)
  
  # drop first row and column and store gene symbols
  cnt_data_filt <- cnt_data_filt[2:nrow(cnt_data_filt),]
  gene_symbols <- cnt_data_filt$hgnc_symbol
  cnt_data_filt <- cnt_data_filt[,1:(ncol(cnt_data_filt)-1)]
  
  cnt_data_low <- cnt_data_low[2:nrow(cnt_data_low),]
  cnt_data_low <- cnt_data_low[,1:(ncol(cnt_data_low)-1)]
  
  paxgene_data_filt <- paxgene_data_filt[,1:(ncol(paxgene_data_filt)-1)]
  
  paxgene_data_low <- paxgene_data_low[,1:(ncol(paxgene_data_low)-1)]
  
  cnt_data_filt_symbols <- cnt_data_filt
  cnt_data_filt_symbols$hgnc_symbol <- gene_symbols
  write.csv(cnt_data_filt_symbols, paste(data_path, paste("processed/", paste(results_prefix, "plasma_cnts.csv", sep="_"), sep=""), sep = ""))
  
  cnt_data_low_symbols <- cnt_data_low
  cnt_data_low_symbols$hgnc_symbol <- gene_symbols
  write.csv(cnt_data_low_symbols, paste(data_path, paste("processed/", paste(results_prefix, "plasma_cnts_low.csv", sep="_"), sep=""), sep = ""))
  
  paxgene_data_filt_symbols <- paxgene_data_filt
  paxgene_data_filt_symbols$hgnc_symbol <- gene_symbols
  write.csv(paxgene_data_filt_symbols, paste(data_path, paste("processed/", paste(results_prefix, "paxgene_cnts.csv", sep="_"), sep=""), sep = ""))
  
  paxgene_data_low_symbols <- paxgene_data_low
  paxgene_data_low_symbols$hgnc_symbol <- gene_symbols
  write.csv(paxgene_data_low_symbols, paste(data_path, paste("processed/", paste(results_prefix, "paxgene_cnts_low.csv", sep="_"), sep=""), sep = ""))
  
  write.csv(meta_data_low, paste(data_path, paste("processed/", paste(results_prefix, "metadata_low.csv", sep="_"), sep=""), sep = ""))
  
  #########################
  # DE ANALYSIS
  #########################
  # build DESeq2 object
  meta_data_tmp <- meta_data_filt
  meta_data_tmp$sepsis_cat[meta_data_tmp$sepsis_cat %in% c("1_Sepsis+BldCx+", "2_Sepsis+OtherCx+")] <- "1_Sepsis+BldCx+"
  if (age_sex_model){
    dds <- DESeqDataSetFromMatrix(countData = cnt_data_filt,
                                  colData = meta_data_tmp,
                                  design = ~ sepsis_cat + age_scaled + gender)
    
    dds_paxgene <- DESeqDataSetFromMatrix(countData = paxgene_data_filt,
                                  colData = meta_data_tmp,
                                  design = ~ sepsis_cat + age_scaled + gender)
    
  }else{
    dds <- DESeqDataSetFromMatrix(countData = cnt_data_filt,
                                  colData = meta_data_tmp,
                                  design = ~ sepsis_cat)
    
    dds_paxgene <- DESeqDataSetFromMatrix(countData = paxgene_data_filt,
                                  colData = meta_data_tmp,
                                  design = ~ sepsis_cat)
  }
  
  
  # choose the reference level for the factor of interest
  dds$sepsis_cat <- relevel(dds$sepsis_cat, ref = "4_NO_Sepsis")
  dds_paxgene$sepsis_cat <- relevel(dds_paxgene$sepsis_cat, ref = "4_NO_Sepsis")
  
  # run DESeq
  dds <- DESeq(dds)
  dds_paxgene <- DESeq(dds_paxgene)
  
  # transform data for PCA plot
  vsd <- vst(dds, blind = TRUE)
  vsd_paxgene <- vst(dds_paxgene, blind = TRUE)
  
  # save metadata
  write.csv(assay(vsd), paste(data_path, paste("processed/", paste(results_prefix, "plasma_vsd.csv", sep="_"), sep=""), sep = ""))
  write.csv(assay(vsd_paxgene), paste(data_path, paste("processed/", paste(results_prefix, "paxgene_vsd.csv", sep="_"), sep=""), sep = ""))
  
  # plot a PCA with LCA labels overlaid
  pca_res <- prcomp(t(assay(vsd)), scale. = TRUE, center = TRUE)
  pdf(paste(results_path, paste(results_prefix, "pca_plasma_vst.pdf", sep="_"), sep = ""),
      width = 6, height = 5
  )
  print(autoplot(pca_res, data =
                   meta_data_tmp, colour = "sepsis_cat"))
  dev.off()
  
  pca_res <- prcomp(t(assay(vsd_paxgene)), scale. = TRUE, center = TRUE)
  pdf(paste(results_path, paste(results_prefix, "pca_paxgene_vst.pdf", sep="_"), sep = ""),
      width = 6, height = 5
  )
  print(autoplot(pca_res, data =
                   meta_data_tmp, colour = "sepsis_cat"))
  dev.off()
  
  # save metadata
  write.csv(meta_data_tmp, paste(data_path, paste("processed/", paste(results_prefix, "metadata.csv", sep="_"), sep=""), sep = ""))
  
  # extract results
  res <- results(dds, contrast = c("sepsis_cat", "1_Sepsis+BldCx+", "4_NO_Sepsis"))
  res_paxgene <- results(dds_paxgene, contrast = c("sepsis_cat", "1_Sepsis+BldCx+", "4_NO_Sepsis"))
  
  # add gene symbols
  res$hgnc_symbol <- gene_symbols
  res_paxgene$hgnc_symbol <- gene_symbols
  
  # sort the genes from lowest to highest given adjusted p-values
  res <- res[order(res$padj, decreasing = F), ]
  res_paxgene <- res_paxgene[order(res_paxgene$padj, decreasing = F), ]
  
  # replace NA values with 1s
  res$padj[is.na(res$padj)] <- 1
  sig_results <- data.frame(res[res$padj < fdr_thresh, ])
  
  res_paxgene$padj[is.na(res_paxgene$padj)] <- 1
  sig_results_paxgene <- data.frame(res_paxgene[res_paxgene$padj < fdr_thresh, ])
  
  # save the output as a CSV file
  write.csv(sig_results, paste(results_path, paste(results_prefix, "plasma_DGEA_results.csv", sep="_"), sep = ""))
  write.csv(sig_results_paxgene, paste(results_path, paste(results_prefix, "paxgene_DGEA_results.csv", sep="_"), sep = ""))
  
  # generate a volcano plot
  # only display top 25 gene symbols
  res_df <- data.frame(res)
  res_df$sig <- res_df$padj < fdr_thresh
  pdf(paste(results_path, paste(results_prefix, "plasma_volcano_plot.pdf", sep="_"), sep = ""), width = 5, height = 5)
  p <- ggplot(res_df, aes(log2FoldChange, -log10(pvalue))) +
    geom_point(aes(col = sig)) +
    scale_color_manual(values = c("black", "red")) +
    ggtitle("Volcano Plot") +
    theme(legend.position = "none") +
    geom_text_repel(data = res_df[1:25, ],
                    aes(label = res_df[1:25, "hgnc_symbol"]))
  print(p)
  dev.off()
  
  res_paxgene_df <- data.frame(res_paxgene)
  res_paxgene_df$sig <- res_paxgene_df$padj < fdr_thresh
  pdf(paste(results_path, paste(results_prefix, "paxgene_volcano_plot.pdf", sep="_"), sep = ""), width = 5, height = 5)
  p <- ggplot(res_paxgene_df, aes(log2FoldChange, -log10(pvalue))) +
    geom_point(aes(col = sig)) +
    scale_color_manual(values = c("black", "red")) +
    ggtitle("Volcano Plot") +
    theme(legend.position = "none") +
    geom_text_repel(data = res_paxgene_df[1:25, ],
                    aes(label = res_paxgene_df[1:25, "hgnc_symbol"]))
  print(p)
  dev.off()
} 
  
