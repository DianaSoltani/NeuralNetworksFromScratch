import numpy as np

data = [
    # Class A
    ([1, 1, 0, 0, 0,
        0, 1, 1, 0, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        0, 0, 0, 0, 0], 0),
    ([1, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        1, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,], 0),
    ([0, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,], 0),
    ([1, 1, 0, 0, 0,
        1, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,], 0),
    ([0, 1, 1, 0, 0,
        1, 1, 0, 0, 1,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,], 0),
    # Class B
    ([0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1,
        0, 0, 1, 1, 1,], 1),
    ([0, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 1,], 1),
    ([0, 1, 0, 0, 0,
        0, 0, 0, 1, 0,
        1, 0, 0, 1, 0,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 0,], 1),
    ([1, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 1,
        0, 1, 0, 1, 0], 1),
    ([0, 0, 0, 0, 0,
        0, 1, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 0,
        0, 0, 1, 1, 1,], 1)
]
mystery = [
    [1, 0, 0, 0, 0,
     1, 0, 1, 0, 1,
     0, 0, 0, 0, 0,
     0, 0, 1, 1, 1,
     0, 0, 0, 1, 0,],

    [1, 1, 1, 0, 0,
     1, 1, 1, 0, 0,
     0, 0, 1, 1, 0,
     0, 0, 0, 0, 0,
     1, 0, 1, 0, 0],

    [0, 0, 0, 0, 1,
     0, 0, 0, 0, 1,
     0, 0, 0, 1, 1,
     0, 0, 0, 0, 1,
     0, 1, 0, 1, 1],

    [0, 1, 1, 0, 0,
     1, 1, 0, 0, 0,
     0, 1, 1, 0, 0,
     0, 1, 0, 0, 1,
     0, 0, 0, 0, 0],

    [1, 0, 0, 0, 1,
     0, 0, 0, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 0, 0, 0,
     1, 0, 0, 0, 1],
]

# Perceptrons for Binary Classification
class Perceptron:
    def __init__(self, inputLength, weights = None):
        # Initialize weights randomly near 0 as an initial guess 


        # For a perceptron, a learning rate is not needed. 
        # Assign learning rate to 1

        

    def update(self, inputData):
        #Iterate through the data

            # Access the weights, inputs, and labels
          

            # Call the activation function
           

            # Regularization done here
            

            # Update the weights
            

    def train(self, inputData):
        # Choose some data point (x,y) from the set of all given Data.
        
            # Update the parameters based on how poorly the model 
            # performs on the specific data point
            

    # Activation Function
    def guess(self, x):
        
'''
    def printWeights(self):
        print(self.weights)
'''


def mainPerceptron():
    
    #print("Correct Answers to the test set: 1 0 1 0 1||0")
    
    #Instantiate an instance of the perceptron class
    p = Perceptron(25)
    
    #Train the perceptron based on the training data
    p.train(data)

    #Now run the model on the testing data and output the results
    for item in mystery:
        if p.guess(item) is 0:
            print("Class A")
        else:
            print("Class B")


if __name__== "__main__":
    print("Perceptron")
    mainPerceptron()