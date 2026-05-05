# FMCW Radar Data Processing Scripts

This repository contains an FMCW radar processing pipeline for DDMA MIMO ADC data acquired from the TI AWR2944EVM using the DCA1000EVM capture module.

The public demo code currently covers raw ADC loading, interference mitigation, range-Doppler map generation, CFAR detection, DBSCAN clustering, tracking, and feature extraction.

It assumes a single track per frame and supports stationary radars only.

> This repository is under active development, so the documentation may not always fully reflect the current code status.

## Dataset

A small example raw FMCW radar ADC dataset is available from the repository Releases page:

[Download the raw DDMA MIMO FMCW radar ADC dataset] https://github.com/mark-098/ddma-fmcw-signal-chain/releases/tag/v1.0 

The dataset is split by radar configuration/profile into separate ZIP files. Each profile contains measurements for human (`HUM`), dog (`DOG`), and no-target/noise-only recordings (`NOS`).


## Schematic overview 
High level overview of the signal chian is presented in the below figure.
![DDMA FMCW radar processing chain](/images/gitProjectStructure_h.png)


## Demo usage

The current demo output control is though basic ```boolean``` parameteres inside the ```ddma_chain_main.py``` file's ```if __name__ == "__main__":``` section.

In addtiion the capture settings and achived velocity and range resolutions must be configured there.

Processing settings, such as:

*   Interference mitigation control,
*   CFAR control,
*   DBSCAN settings,
*   KF tracker,

are controlled from within the ```ddma_dataclasses.py``` file.
