import pickle
import os, sys
class model:
    def __init__(self, shared=""):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_labels=1
        self.is_trained=False
        self.shared=shared

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...


        '''
        
        #self.num_train_samples = X.shape[0]
        self.is_trained=True
        print("Training is done!")

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
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

