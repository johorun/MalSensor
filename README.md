README

# MalSensor

Fast and Robust Windows Malware Classification.


## Setup

* `IDA Pro`: >= 7.5

## Dataset

The dataset `MalwareBazaar` `MalwareDrift` and label files `MalwareBazaar_Labels.csv` `MalwareBazaar_Labels.csv` used in this paper and code come from the following paper.

\[FSE2021\] [A Comprehensive Study on Learning-Based PE Malware Family Classification Methods.](https://dl.acm.org/doi/abs/10.1145/3468264.3473925)

Dataset:<https://github.com/MHunt-er/Benchmarking-Malware-Family-Classification>




## Useage

Disassembly


Put the `auto_script.py` `static_analysis.py` in the IDA Pro working directory. 

Run `python auto_script.py`   `-d`  Malware_path   `-i`  IDA_path


Generate function call graph gexf

Run `python APIextract.py` to deduplicate the `imp_API.txt` file and generate `APIlist.txt` file

Put the `.info` files of malware generated by disassembly 

Run `python cg.py` to generate the function call graph of each sample.


FeatureExtraction

Run `python FeatureExtraction.py`  `-d`  Gexf_path  `-o`  Output_path `-c`  Centrality_type 

Then, run `python transtomulti.py` to generate the feature csv file with sample feature vectors and label information according to the centrality.


Classification

Run  `python Classification.py` `-d` features_csv_path `-o` Output_path `-t` Centrality_type

