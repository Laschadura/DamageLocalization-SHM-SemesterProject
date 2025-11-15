# Damage Localization on a Masonry Arch Bridge (Semester Project, ETH Zürich)

This repository contains the code from my semester project on **damage detection and localization for structural health monitoring** of a masonry arch bridge.

## Structure

- `MATLAB/`  
  Modal parameter extraction (SSI, FDD, ERA) from accelerometer data.

- `Modal_Parameter_Model/`  
  Machine learning models that use extracted modal parameters for **damage classification and localization**.

- `RawDataModel/`  
  Deep learning models (e.g. 1D CNNs) that work directly on raw or preprocessed time series data.

- `report/SP-Report_SHM_Simon_Scandella.pdf`  
  Project report describing the experiment, methods, and results.

> Note: The original measurement data cannot be shared publicly, so this repository only contains code and documentation.
