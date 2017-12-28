# Anytime-SCAN
Source Code for the Anytime-SCAN paper

## 1. Datasets Downloading
The datasets used in this paper is publicly avlaible in DropBox [AnySCAN](https://www.dropbox.com/sh/6anjkvdss8k46t2/AACjox26kmBsvVKK3cS7jra0a?dl=0).
The format of the dataset file is edge list, each edge like "node1 node2" per line

## 2. Run the Code
To run the code, the library "networkx" needs to be installed on Python 2.7 or 3.x.
AnySCAN and original SCAN are both implemended in this code.

### a). Orignial SCAN
To run the original SCAN algorithm on the input dataset, use the command: python AnytimeSCAN.py edge-list-file-name scan epsilon(float) mu(int).

The clustering result file entitled "scan_results_node_info_[dataset-name]_[epsilon]_[mu].txt" will be output in the current working directory.

If you have the ground truth of the dataset (with the format "node_ID cluster_ID" per line, hubs with cluster_ID -1 and outliers with cluster_ID -2), to calculate the NMI (Normalized Mutual Information) and ARI (Adjusted Rand Index), use the command: python AnytimeSCAN.py edge-list-file-name scan epsilon(float) mu(int) true_label_filename.

The clustering result file entitled "scan_results_node_info_[dataset-name]_[epsilon]_[mu].txt" will be output in the current working directory. The NMI and ARI will be output in the console.

### b). AnySCAN




