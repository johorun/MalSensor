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



