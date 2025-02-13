# CL & CALIFA Pipeline

This repository contains a  pipeline for processing and analyzing data from the CL and CALIFA surveys. The pipeline includes:

- **Training a SimSiam model:** The `train.py` script sets up and trains a contrastive learning model.
- **Generating galaxy pairs:** The `generate_galaxy_pairs.py` script creates augmented galaxy pairs from FITS data.
- **Creating latent space projections:** The `create_latent_space.py` script computes latent representations (projections) from a trained model.
- **Merging TFRecord files:** The `merge_trecords.py` script merges multiple TFRecord files into consolidated files.

## Directory Structure
