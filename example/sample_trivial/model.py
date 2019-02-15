import pickle
import os, sys
class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_labels=1
        self.is_trained=False

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        
        #self.num_train_samples = X.shape[0]
        self.is_trained=True
        print("Training is done!")

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
	In out case X is a dictionary of two fields: Fermi Image and its WCS object.

        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        y = []
        return [[[0,0]] for _ in range(len(X))]
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"), 2) 

    def load(self, path="./"):
        print(path)
        modelfile = path + '_model.pickle'
        print(modelfile)
        if os.path.isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

