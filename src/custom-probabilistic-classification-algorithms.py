import numpy as np



class GaussianBayes:
    ''' Implements the Gaussian Bayes For Classification without assuming feature independence. '''

    def __init__(self):
        self.means = {}
        self.covariances = {}
        self.priors = {}
        self.classes = []

    def train(self, X, Y):
        ''' Train the multiclass (or Binary) Bayes Rule using the given 
            X [m x n] data matrix and Y labels matrix'''

        # Getting the unique classes
        self.classes = np.unique(Y)
        m, n = X.shape

        for c in self.classes:
            # Selecting the data points belonging to class c
            X_c = X[Y == c]

            # Calculating the mean for each feature in class c
            self.means[c] = np.mean(X_c, axis=0)
            
            # Calculating the covariance matrix for class c
            self.covariances[c] = np.cov(X_c, rowvar=False)
            
            # Calculating the prior probability for class c
            self.priors[c] = X_c.shape[0] / m

    def test(self, X):
        ''' Run the trained classifiers on the given set of examples 
            For each example, you should return probability and its assigned class
            Input: X of m x d
            Output:
            pclasses: predicted class of each example
            probabilities: probability of each example falling in that predicted class...
        '''

        m, d = X.shape
        pclasses = []
        probabilities = np.zeros(m)

        for i in range(m):
            # Initializing max probability and class
            max_prob = -1
            best_class = None
            
            for c in self.classes:
                mean = self.means[c]
                cov = self.covariances[c]
                prior = self.priors[c]
                
                # Calculating the multivariate Gaussian probability P(X|C)
                cov_inv = np.linalg.inv(cov)  # Inverse of covariance matrix
                cov_det = np.linalg.det(cov)  # Determinant of covariance matrix
                diff = X[i] - mean
                
                likelihood = (1 / np.sqrt((2 * np.pi) ** d * cov_det)) * \
                             np.exp(-0.5 * np.dot(np.dot(diff.T, cov_inv), diff))
                
                # Calculating the posterior probability P(C|X)
                posterior = likelihood * prior
                
                # Updating max probability and class
                if posterior > max_prob:
                    max_prob = posterior
                    best_class = c
            
            # Assigning the best class and its probability
            pclasses.append(best_class)
            probabilities[i] = max_prob
        
        return pclasses, probabilities

    def predict(self, X):
        ''' Predicts the class for each example in X '''
        
        return self.test(X)[0]
    
class GaussianNaiveBayes:
    ''' Implements the Gaussian Naive Bayes for Classification '''
    
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.priors = {}
        self.classes = []
    
    def train(self, X, Y):
        ''' Train the Gaussian Naive Bayes model using X (features) and Y (labels) '''
        
        # Getting unique classes from the dataset
        self.classes = np.unique(Y)
        m, n = X.shape
        
        for c in self.classes:
            # Selecting data points for class c
            X_c = X[Y == c]

            # Calculating mean and variance for each feature in class c
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)

            # Calculating prior probability for class c
            self.priors[c] = X_c.shape[0] / m
        
    def test(self, X):
        ''' Test the Gaussian Naive Bayes model on input data X
            Returns the predicted class and probability for each example '''
        
        m, d = X.shape
        pclasses = []
        probabilities = np.zeros(m)

        for i in range(m):
            max_prob = -1
            best_class = None
            
            for c in self.classes:
                # Fetching mean, variance, and prior for class c
                mean = self.means[c]
                var = self.variances[c]
                prior = self.priors[c]
                
                # Calculating the Gaussian likelihood for each feature
                likelihood = np.prod(
                    (1 / np.sqrt(2 * np.pi * var)) * 
                    np.exp(-((X[i] - mean) ** 2) / (2 * var))
                )
                
                # Calculating the posterior probability P(C|X)
                posterior = likelihood * prior
                
                # Checking if this is the best class so far
                if posterior > max_prob:
                    max_prob = posterior
                    best_class = c
            
            # Storing the best class and its probability
            pclasses.append(best_class)
            probabilities[i] = max_prob
        
        return pclasses, probabilities
    
    def predict(self, X):
        ''' Predicts the class for each example in X '''
        
        return self.test(X)[0]
    
class KNearestNeighbors:
    ''' Implements the K-Nearest Neighbors (KNN) algorithm for classification '''

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, Y):
        ''' Store the training data for the KNN classifier '''

        self.X_train = X
        self.y_train = Y

    def euclidean_distance(self, x1, x2):
        ''' Calculate the Euclidean distance between two points '''
        
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def classify(self, X_test_instance):
        ''' Classify a single test instance using the KNN algorithm '''

        # Computing all Euclidean distances between X_test_instance and X_train
        distances = np.array([self.euclidean_distance(X_test_instance, x_train_instance) 
                              for x_train_instance in self.X_train])

        # Finding the indices of the k smallest distances
        k_nearest_indices = np.argsort(distances)[:self.k]

        # Getting the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_nearest_indices]

        # Performing majority voting
        unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)

        # Returning the label with the highest count (vote)
        majority_vote_label = unique_labels[np.argmax(label_counts)]

        return majority_vote_label

    def test(self, X):
        ''' Predict the class for each instance in X '''

        predictions = [self.classify(X_test_instance) for X_test_instance in X]

        return np.array(predictions)

    def predict(self, X):
        ''' Alias for the test method '''

        return self.test(X)

