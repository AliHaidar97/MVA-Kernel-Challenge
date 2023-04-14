import pandas as pd
import pickle as pkl 
import compute_results
import argparse
import utils

# Defining main function
def main(precomputed = 0):
    
    path = 'data/'
    with open(path + 'training_data.pkl', 'rb') as file: 
        train_graphs = pkl.load(file) 

    with open(path + 'test_data.pkl', 'rb') as file: 
        test_graphs = pkl.load(file) 
        
    with open(path + 'training_labels.pkl', 'rb') as file: 
        train_labels = pkl.load(file) 
        
    # Call the 'fix_graphs' function from the 'utils' module to preprocess the graphs.
    train_graphs, test_graphs = utils.fix_graphs(train_graphs, test_graphs)
    
    # Call the 'morgan_index' function from the 'utils' module to compute Morgan fingerprints
    # for the training and testing graphs.
    for i in range(1):
        train_graphs = utils.morgan_index(train_graphs)
        test_graphs = utils.morgan_index(test_graphs)
        
            
    sub_full_dataset = compute_results.compute_results_full_dataset(train_graphs.copy(), test_graphs.copy(), train_labels.copy(), precomputed)
    sub_full_dataset.to_csv("new_submissions/sub_full_dataset.csv", index=False)
    
    if(precomputed == 0):
        
        sub_bootstrap = compute_results.compute_results_bootstrap(train_graphs.copy(), test_graphs.copy(), train_labels.copy())
        sub_bootstrap.to_csv("new_submissions/sub_bootstrap.csv", index=False)
        
        sub_merge = sub_full_dataset.copy()
        sub_merge['Predicted'] += sub_bootstrap['Predicted']
        sub_merge['Predicted'] /= 2
        sub_merge.to_csv("new_submissions/sub_merge.csv", index=False)
        
if __name__=="__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-p", "--precomputed", help = "Use precomputed kernel", default = 0)
    
    # Read arguments from command line
    args = parser.parse_args()
    
    main(args.precomputed)