import numpy as np
import logging
from tqdm import tqdm


class Perceptron:
    '''
    Functions in this class:
        * main function
        * activationFunction
        * fit function
        * predict 
        * total_loss
    '''
    
    def __init__(self, eta, epochs):
        # eta - learning rate
        # epoch = forward + backward
        
        self.weights = np.random.randn(3) * 1e-4 # SMALL WEIGHT INITIALIZATION
        logging.info(f"initial weights before training: {self.weights}")
        self.eta = eta # LEARNING RATE
        self.epochs = epochs
    
    def activationFunction(self, inputs, weights):
        z = np.dot(inputs, weights) # z = X * W
        return np.where(z > 0, 1, 0)
    
    def fit(self, X, y):
        self.X = X
        self.y = y 
        
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))] # CONCAT THE X ARRAY WITH THE NEW ARRAY CONSISTING OF SIZE (SIZE_OF_X_ARRAY, 1)
                                                                # CREATING A NEW ARRAY i.e array with the corresponding bias and concatinating to X array 
        logging.info(f"X with bias : \n{X_with_bias}")
        
        #  TRAINING LOOP
        for epoch in tqdm(range(self.epochs), total = self.epochs, desc="traingin the models"):
            logging.info('--'*10)
            logging.info(f"for epoch: {epoch}")
            logging.info('--'*10)
            
            '''FORWARD PROPAGATION'''
            y_hat = self.activationFunction(X_with_bias, self.weights) # y_hat -> Predicated value
            logging.info(f"Predicted value after forward pass: {y_hat}")
            
            '''ERROR'''
            self.error = self.y - y_hat
            logging.info(f"Error: \n{self.error}")
            
            '''BACKWARD PROPAGATION'''
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            logging.info(f"Updated weights after epoch: {epoch}/{self.epochs} : {self.weights}")
            
            logging.info("#####"*10)
    
    def predict(self, X):
        X_with_bias = np.c_[X, -np.ones((len(X), 1), dtype=int)]
        return self.activationFunction(X_with_bias, self.weights)
    
    def total_loss(self):
        total_loss = np.sum(self.error)
        logging.info(f"Total Loss: {total_loss}")
        return total_loss