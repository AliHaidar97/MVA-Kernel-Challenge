# Graph Classification with Kernel Methods
This code is a part of a project that uses kernel methods to classify graphs. The main function of this code loads the training and test data, and computes the classification results using kernel methods.

## Usage
To use this code, run the following command:
python main.py [-h] [-p PRECOMPUTED]
The optional argument -p or --precomputed can be used to specify whether to use precomputed kernel or not.
The default value is 0, which means that the kernel will be computed during the run.
if you use precomputed, please ensure that the precomputed kernels exist in precomputed_kernels folder. (The data is larger than 1GB)

## Input
This code requires the following input files in the data/ directory:

- training_data.pkl: A pickle file containing the training data.
- test_data.pkl: A pickle file containing the test data.
- training_labels.pkl: A pickle file containing the labels of the training data.
## Output
This code generates the following output files in the new_submissions/ directory:

- sub_full_dataset.csv: The classification results for the full dataset.
- sub_bootstrap.csv: The classification results for the bootstrap dataset.
- sub_merge.csv: The classification results obtained by merging the results of the full dataset and the bootstrap dataset.
