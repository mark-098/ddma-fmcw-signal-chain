# FMCW Radar Data Processing Scripts

This repository contains an FMCW radar processing pipeline for DDMA MIMO ADC data acquired from the TI AWR2944EVM using the DCA1000EVM capture module.

The public demo code currently covers raw ADC loading, interference mitigation, range-Doppler map generation, CFAR detection, DBSCAN clustering, tracking, and feature extraction.

It assumes a single track per frame and supports stationary radars only.

> This repository is under active development, so the documentation may not always fully reflect the current code status.

## Datasets
Data acquiered with the DCA1000EVm and the AWR2944EVM may which are to be used within this demo may be found under: [LINK]


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
