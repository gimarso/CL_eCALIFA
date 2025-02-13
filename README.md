# Exploring Galaxy Properties of eCALIFA with Contrastive Learning

This repository contains a pipeline for processing and analyzing data with **Contrastive Learning (CL) on CALIFA surveys**. 

## Features
The pipeline includes:

- **Training a SimSiam model**  
  The `train.py` script sets up and trains a contrastive learning model to learn representations of galaxy images.
  
- **Generating galaxy pairs**  
  The `generate_galaxy_pairs.py` script creates augmented galaxy pairs from CALIFA FITS cubes for contrastive learning.

- **Creating latent space projections**  
  The `create_latent_space.py` script computes **latent representations (embeddings)** of galaxies and their properties.

- **Merging TFRecord files**  
  The `merge_trecords.py` script merges multiple TFRecord files into consolidated files.

---

## **Results**
Here are some of the results obtained from the analysis.

### **1️⃣ Original & Transformed Galaxy Images**  
This image shows the original galaxy and its transformed version produced by the pipeline.

<p align="center">
  <img src="images/original_transform.png" alt="Original and transformed galaxy images" width="600">
</p>

### **2️⃣ Dimensionality Reduction with PCA**  
Principal Component Analysis (PCA) is applied to reduce the dimensionality of CALIFA datacubes.

<p align="center">
  <img src="images/dim_reduction.png" alt="PCA dimensionality reduction of CALIFA datacubes" width="600">
</p>

### **3️⃣ UMAP Projection of Latent Space**  
The UMAP projection visualizes the learned latent space from the **SimSiam model**.

<p align="center">
  <img src="images/embbeding_projections.png" alt="UMAP projection of SimSiam embedding space" width="600">
</p>

---

## **Training Visualization**
The following video shows the training process of the SimSiam model.

<p align="center">
  <a href="https://youtu.be/D6EdMDz58Qw">
    <img src="images/video_preview.png" alt="Click to watch training video" width="600">
  </a>
</p>

---

## **How to Use This Pipeline**
To train and analyze the data, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/gimarso/CL_eCALIFA.git
   cd CL_eCALIFA