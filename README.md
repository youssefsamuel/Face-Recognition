# Face-Recognition

## Dataset Used
Dataset is available at the following link: https://www.kaggle.com/kasikrit/att-database-of-faces/

## Problem statement
It is required to perform face recognition using PCA and LDA as an algorithm for dimensionality 
reduction, then for classification K-NN algorithm is used.
Two classification problems are required:

• Classification of a test face as one of 40 possible classes.

• Face vs non-face classification problem. Given a test image, is this image a face or a non face?

## Dataset description
There are 10 different images of 40 distinct people. The images were taken at different times, varying lighting slightly, facial expressions (open/closed eyes, smiling/non-smiling) and facial details (glasses/no-glasses).  All the images are taken against a dark homogeneous background and the subjects are in up-right, frontal position (with tolerance for some side movement).
The data matrix is a numpy array of size (400 x 10304), where 400 is the total number of faces because there 40 classes and and 10 images per class. The number of features for each image is 10304.
In case of faces vs non faces classification problem another dataset is used of size (800 x 10304) containing this time 800 images. First, 400 non face then another 400 for faces.

## Main user defined functions
### Knn
The knn function is used to classify samples in a testing set using the k-nearest neighbors algorithm.
Variables:
•	projected_testing_set: The testing set data that has been projected onto a lower-dimensional space.
•	projected_training_set: The training set data that has been projected onto the same lower-dimensional space as projected_testing_set.
•	number_of_classes: An integer representing the number of classes in the dataset.
•	training_labels: The corresponding class labels for the samples in projected_training_set.
•	euc_dist_matrix: A numpy array representing the pairwise Euclidean distances between each sample in the testing set and each sample in the training set.
•	k: An integer representing the number of nearest neighbors to consider for classification.
The function loops through each sample in projected_testing_set and performs the following steps:
1.	Finds the k nearest neighbors in projected_training_set to the current sample in projected_testing_set.
2.	Computes the weighted votes of each class based on the distances to the k nearest neighbors.
3.	Determines the class with the highest weighted vote.
4.	Assigns the class label of the testing sample to the winning class.
5.	Returns an array containing the predicted class labels for each sample in the testing set.
Note: training_labels are assumed to start from 1, so training_labels[id_index] - 1 is used to convert the label to a zero-based index.

### PCA
The function performs Principal Component Analysis (PCA) on the input dataset to reduce the dimensionality of the feature space and then applies K-Nearest Neighbor (KNN) classification to identify the class of each test data point. The function also provides an option to display the results using confusion matrices and/or images.
Inputs:
•	D: The dataset matrix
•	label_vector: The label vector corresponding to each data point in the dataset matrix
•	alpha: The percentage of variance to retain after PCA, can be a single value or a list of values
•	train_portion: The portion of the dataset to be used for training, if normal_split=False
•	n_classes: The number of classes in the dataset
•	n_perclass: The number of samples per class in the dataset
•	k_knn: The number of nearest neighbors to consider for classification, can be a single value or a list of values
•	normal_split: A boolean variable indicating whether to use the normal dataset split or not
Outputs:
•	accuracy: The accuracy of the classification for each value of alpha and k_knn
•	cm: The confusion matrix of the classification results
Steps:
1.	Splitting the input dataset into training and testing sets based on the provided parameters.
2.	Computing the mean vector and centering the training set around it.
3.	Calculating the covariance matrix of the centered training set.
4.	Computing the eigenvalues and eigenvectors of the covariance matrix.
5.	Sorting the eigenvalues and eigenvectors in descending order of eigenvalues.
6.	Selecting the top r eigenvectors based on a threshold alpha value provided as input.
7.	Projecting the training and testing sets onto the selected eigenvectors.
8.	Computing the Euclidean distance between each testing set image and every training set image in the projected space.
9.	Classifying each testing set image by performing k-NN classification using the k nearest training set images in the projected space.
10.	Evaluating the classification accuracy of the model for each combination of alpha and k values provided as input.
11.	Displaying the confusion matrix of the final classification results and, if requested, displaying some sample images with their true and predicted labels.
12.	Returning the computed accuracy and confusion matrix as output.

### LDA
The function lda performs Linear Discriminant Analysis (LDA) on a given dataset D with corresponding label_vector. It returns the classification accuracy and confusion matrix for the testing set.
The function has the following parameters:
•	D: an array of size (number of data samples) x (number of features), representing the dataset
•	label_vector: an array of size (number of data samples) x 1, representing the class labels for each data sample
•	train_portion: a float representing the proportion of the data to use for training (default is 0.5)
•	n_classes: an integer representing the number of classes in the dataset
•	n_perclass: an integer representing the number of data samples per class
•	n_eig_used: an integer representing the number of eigenvalues to use in the projection (default is all)
•	k_knn: an integer or a list of integers representing the number of nearest neighbors to consider in the K-NN algorithm (default is 1)
Steps:
1.	Splitting the dataset into training and testing sets using one of three splitting methods based on the input parameters: split_nonfaces, split_even_odd, or split.
2.	Computing the mean vector of the training set and centering the data by subtracting the mean from each data point.
3.	Computing the covariance matrix of the centered training data.
4.	Computing the eigenvalues and eigenvectors of the covariance matrix.
5.	Sorting the eigenvalues in descending order and selecting the eigenvectors corresponding to the r_alpha largest eigenvalues, where r_alpha is determined by the input parameter alpha.
6.	Projecting the centered training and testing data onto the selected eigenvectors.
7.	Computing the Euclidean distance between each test data point and all training data points in the projected space.
8.	Classifying each test data point using k-nearest neighbors algorithm based on the input parameter k_knn.
9.	Computing the confusion matrix and accuracy of the classification.
10.	Displaying the images and classification results if the input parameter normal_split is True.
11.	Returning the accuracy and confusion matrix as output.

## Sample Runs performed in notebook
1. PCA and accuracies for different values of alpha
2. Kernel PCA and comparison with PCA
3. LDA
4. QDA
5. PCA vs LDA
6. QDA vs LDA
7. Faces vs Non Faces Classification




