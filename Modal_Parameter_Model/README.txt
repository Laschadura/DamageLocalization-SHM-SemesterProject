
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Folders:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

0_old: Old scripts/old versions

Clustered Results: The clustered ERA dataset (with Cluster ID) used in Image_Model_V0.py

Labels: Folder containing the image labels

ModalParameters & ModeShapes: SSI Dataset

Results ERA: The ERA script output. Isn't used in any pipeline here since we use the clustered version for Image_Model_V0.py

Results FDD: FDD dataset used in RF_simpleCNN_FDD_Dataset.py

ResultsPlotsPipelines: Containes plots and visualizations of each pipeline

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Files:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Damage_Labels_10.csv & Damage_Labels_20.csv: Zone based labels used in the SSI and FDD pipelines


FDD_find_reference_modes.py: Script which exports reference_modes.csv used in the FDD pipeline

reference_modes.csv: Reference modes used in the FDD pipeline


SSI_find_reference_modes.py: Script which exports final_reference_modes.csv used in the SSI pipeline (This loads Selection.csv)

final_reference_modes.csv: Reference modes used in the SSI pipeline

Selection.csv: Manually selected mode shapes from SSI dataset used to find final_reference_modes.csv


-> Pipelines:

SSI_pipeline_RF_MLP.py: SSI pipeline

FDD_pipeline_RF_MLP_CNN_CNN-SE.py: FDD pipeline

Image_Model_V0.py: ERA pipeline

