# FMCW Radar Data Processing Scripts

This repository contains an FMCW radar processing pipeline for DDMA MIMO ADC data acquired from the TI AWR2944EVM using the DCA1000EVM capture module.

The public demo code currently covers raw ADC loading, interference mitigation, range-Doppler map generation, CFAR detection, DBSCAN clustering, tracking, and feature extraction.

> This repository is under active development, so the documentation may not always fully reflect the current code status.


## Signal Chain

The full graphical overview is available in the Colab notebook:

[Open repository overview notebook](https://github.com/mark-098/ddma-fmcw-signal-chain/blob/main/notebooks/RepoDescription.ipynb)
