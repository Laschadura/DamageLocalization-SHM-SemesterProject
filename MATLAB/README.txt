The scripts here perform modal parameter extraction on the raw accelerometer data provided by the Leeds team

DATA:
- Contains the accelerometer data from the Leeds team: VibrationalTestData

- Contains the Preprocessed Data with augmentation for the SSI script: PreprocessedData
- Contains the output of the SSI_script: ModalParameters

Results:
- Outputfolder for the ERA and FDD scripts

Source:
- Contains the functions needed to use the scripts.
Each script provides details in the comments on which Sources files it needs.

Clustering Structural Modes:
- Contains scripts which use the ERA Data or FDD Data and cluster the modes across different tests

Scripts:
- ERA_script: Performs ERA on the raw data
- Plot_Modes: Plots the modes (2D/3D)
- Preprocessing_Data_wAugmentation: Augments the data with noise and produces for each CSV file 5 additional augmented files