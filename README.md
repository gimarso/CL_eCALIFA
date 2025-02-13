# CL & CALIFA Pipeline

This repository contains a  pipeline for processing and analyzing data with CL on CALIFA surveys. The pipeline includes:

- **Training a SimSiam model:** The `train.py` script sets up and trains a contrastive learning model.
- **Generating galaxy pairs:** The `generate_galaxy_pairs.py` script creates augmented galaxy pairs from CALIFA FITS cubes.
- **Creating latent space projections:** The `create_latent_space.py` script computes latent representations (projections) from a trained model and galaxy properties.
- **Merging TFRecord files:** The `merge_trecords.py` script merges multiple TFRecord files into consolidated files.

## Results

![Original and transform version produced by the pipeline ](images/original_transform.png)
![Dimensionality reduction of CALIFA datacubes using PCA](images/dim_reduction.png)
![UMAP projection of the ebbeding obtained with SimSiam model](images/embbeding_projections.png)


## Training Visualization

[![Watch the video](images/video_preview.png)](videos/embedding_evolution_3d.mp4)

## Directory Structure
