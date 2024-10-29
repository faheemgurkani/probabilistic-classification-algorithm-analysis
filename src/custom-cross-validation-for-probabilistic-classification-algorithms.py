import numpy as np



class KFoldCV:

    def __init__(self, k=5):
        self.k = k

    def split(self, X, y):
        """
        Splits the data into k folds.
        """

        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        fold_size = num_samples // self.k
        folds = []
        
        for i in range(self.k):
            start_index = i * fold_size
            end_index = start_index + fold_size if i != self.k - 1 else num_samples
            fold_indices = indices[start_index:end_index]
            folds.append(fold_indices)

        return folds

    def train_test_split(self, X, y, fold):
        """
        Splits the data into training and testing sets based on the fold.
        """

        test_indices = fold
        train_indices = np.array([i for i in range(X.shape[0]) if i not in test_indices])

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        return X_train, y_train, X_test, y_test

    def cross_validate(self, X, y, model):
        """
        Performs k-fold cross-validation on the given model.
        """
        folds = self.split(X, y)
        scores = []

        for fold in folds:
            X_train, y_train, X_test, y_test = self.train_test_split(X, y, fold)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scores.append(np.mean(predictions == y_test))

        return np.mean(scores), np.std(scores)

class StratifiedKFoldCV:

    def __init__(self, k=5):
        self.k = k

    def split(self, X, y):
        """
        Splits the data into k stratified folds, ensuring that each fold has
        a similar distribution of class labels as the original dataset.
        """
    
        num_samples = X.shape[0]
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        
        # Shuffling class indices to randomize the data within each class
        for cls in unique_classes:
            np.random.shuffle(class_indices[cls])
        
        folds = [[] for _ in range(self.k)]
        
        # Distributing samples across folds while maintaining class proportions
        for cls in unique_classes:
            cls_indices = class_indices[cls]
            fold_size = len(cls_indices) // self.k
            
            for i in range(self.k):
                start_index = i * fold_size
                end_index = start_index + fold_size if i != self.k - 1 else len(cls_indices)
                fold_indices = cls_indices[start_index:end_index]
                folds[i].extend(fold_indices)
        
        folds = [np.array(fold) for fold in folds]
        
        return folds

    def train_test_split(self, X, y, fold):
        """
        Splits the data into training and testing sets based on the fold.
        """
        
        test_indices = fold
        train_indices = np.array([i for i in range(X.shape[0]) if i not in test_indices])

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        return X_train, y_train, X_test, y_test

    def cross_validate(self, X, y, model):
        """
        Performs stratified k-fold cross-validation on the given model.
        """
        
        folds = self.split(X, y)
        scores = []

        for fold in folds:
            X_train, y_train, X_test, y_test = self.train_test_split(X, y, fold)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scores.append(np.mean(predictions == y_test))

        return np.mean(scores), np.std(scores)


