import utils
import models
import numpy as np
import kernels.ShortestPathKernel as  ShortestPathKernel
import kernels.WalkKernel as WalkKernel
import pandas as pd

# Define a function to compute results for the full dataset.
# It takes three arguments:
# - train_graphs: a list of graphs to use for training.
# - test_graphs: a list of graphs to use for testing.
# - train_labels: a list of labels for the training graphs.
def compute_results_full_dataset(train_graphs, test_graphs, train_labels, precomputed = 0):
    
    # Create instances of the WalkKernel and ShortestPath classes.
    walkKernel = WalkKernel.WalkKernel(maxK=100)  
    shortestPath = ShortestPathKernel.ShortestPath()

    # Compute the kernel matrices for the training and testing graphs using the Shortest Path and Walk kernels.
    # For the training set:
    if(precomputed == 0):
        print("----------Train----------")
        print("----------Compute Shortest Path Kernel----------")
        K_train = shortestPath.compute_kernel(train_graphs, train_graphs)
        print("----------Compute Walk Kernel----------")  
        K_train += walkKernel.compute_kernel(train_graphs, train_graphs)
        
        # For the testing set:
        print("----------Test----------")
        print("----------Compute Shortest Path Kernel----------")
        K_test = shortestPath.compute_kernel(test_graphs, train_graphs)
        print("----------Compute Walk Kernel----------")  
        K_test += walkKernel.compute_kernel(test_graphs, train_graphs)
    
    else:
        print("----------Load Kernels----------")
        K_train = np.loadtxt('precomputed_kernels/train_morgan_index_1.csv', delimiter=',')
        K_test = np.loadtxt('precomputed_kernels/test_morgan_index_1.csv', delimiter=',')
    
    np.savetxt('precomputed_kernels/train_morgan_index_1.csv', K_train, delimiter=',')
    np.savetxt('precomputed_kernels/test_morgan_index_1.csv', K_test, delimiter=',')
    
    # Preprocess the training labels to -1 or 1.
    y_train = train_labels
    y_train = np.array(y_train).reshape(-1)
    y_train = 2 * y_train - 1 
    
    
    # Create an instance of the KernelSVC class from the 'models' module with a regularization parameter of 1.
    clf = models.KernelSVC(C = 1)
    
    # Fit the model on the training kernel matrix.
    clf.fit(K_train, y_train)

    # Predict the probability of each test graph being in the positive class.
    y_pred = clf.predict_proba(K_test)
    # Convert the predicted probabilities to logarithmic scale.
    y_pred = np.log(y_pred / (1 - y_pred))

    # Create a Pandas DataFrame to store the results.
    sub = pd.DataFrame()
    sub['Id'] = np.arange(1, len(y_pred) + 1)
    sub['Predicted'] = y_pred
    
    # Return the DataFrame with the results.
    return sub




# Define a function to compute results using bootstrap.
# It takes three arguments:
# - train_graphs: a list of graphs to use for training.
# - test_graphs: a list of graphs to use for testing.
# - train_labels: a list of labels for the training graphs.
def compute_results_bootstrap(train_graphs, test_graphs, train_labels):
    
    # Split the training graphs into two lists, one for graphs labeled with 1 and another for graphs labeled with 0.
    one_train = []
    zero_train = []
    for (i, G) in enumerate(train_graphs):
        if train_labels[i] == 0:
            zero_train.append(G)
        else:
            one_train.append(G)
    
    n = len(zero_train) // 9
    chunck_train = [zero_train[i:i + n] for i in range(0, len(zero_train), n)]
    
    # For each chunk, combine it with the list of graphs labeled with 1 to create a new training set.
    # The corresponding training labels are -1 for the graphs in the chunk and 1 for the graphs labeled with 1.
    train_graphs = []
    train_labels = []
    for G in chunck_train:
        train_graphs.append(one_train + G)
        train_labels.append([1] * len(one_train) + [-1] * len(G))
        
    # Create instances of the WalkKernel and ShortestPath classes.
    randomWalk = WalkKernel.WalkKernel(maxK=100)  
    shortestPath = ShortestPathKernel.ShortestPath()
    
    # Compute the kernel matrix for the new training set.
    print("----------Train----------")
    K_train = []
    for i in range(len(train_graphs)):
        print("Chunk " + str(i+1) + "/9")
        K_train.append(shortestPath.compute_kernel(train_graphs[i], train_graphs[i]) + randomWalk.compute_kernel(train_graphs[i], train_graphs[i]))
        
    # Compute the kernel matrix for the testing set.
    print("----------Test----------")
    K_test = []
    for i in range(len(train_graphs)):
        print("Chunk " + str(i+1) + "/9")
        K_test.append(shortestPath.compute_kernel(test_graphs, train_graphs[i]) + randomWalk.compute_kernel(test_graphs, train_graphs[i]))    
    
    # For each chunk, fit a kernel SVM on the corresponding kernel matrix and predict on the test kernel matrix.
    y_pred = []
    for i in range(len(train_graphs)):
        clf = models.KernelSVC(C=1)
        clf.fit(K_train[i], train_labels[i])
        y = clf.predict_proba(K_test[i])
        y_pred.append(y)
    
    # Average the predictions across all the chunks.
    y_pred = np.array(y_pred)
    y_pred = np.mean(y_pred, axis=0)
    y_pred = np.log(y_pred / (1 - y_pred))
    
    # Create a DataFrame to hold the predictions and return it.
    sub = pd.DataFrame()
    sub['Id'] = np.arange(1, len(y_pred) + 1)
    sub['Predicted'] = y_pred
    
    return sub
