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
    def __init__(self, inputLength, generations, weights = None):
        # Initialize w(0) randomly as an initial guess 
        if weights is None:
            self.weights = np.random.random((inputLength))*0.1
        self.learningRate = 1
        self.numGens = generations

    def update(self, inputData):
        for i in range(len(inputData)):
            w = self.weights
            x = inputData[i][0]
            y = inputData[i][1]
            guess = self.guess(x)
            # Regularization Rate
            loss = guess - y

            # Update the weight
            self.weights = np.subtract(w, self.learningRate * np.multiply(loss, x))

    def train(self, inputData):
        # Choose some data point (x,y) from the set of all given Data.
        for _ in range(self.numGens):
            # Update the parameters based on how poorly the model 
            # performs on the specific data point
            self.update(inputData)

    # Activation Function
    def guess(self, x):
        if np.dot(self.weights, x) >= 0.0:
            return 1
        else:
            return 0

    def printWeights(self):
        print(self.weights)

    answers = [1,0,1,0,1]
    meanAcc = 0.0
    for i in range(numPerceptrons):
        p = Perceptron(25, gens)
        p.train(data)
        guess = [p.guess(item) for item in mystery]
        meanAcc += solveAccuracy(answers, guess)

    meanAcc = meanAcc/numPerceptrons
    return meanAcc

def mainPerceptron():
    
    p = Perceptron(25, 1)
    #print("Correct Answers: 1 0 1 0 1|0")
    p.train(data)
    for item in mystery:
        if p.guess(item) is 0:
            print("Class A")
        else:
            print("Class B")

if __name__== "__main__":
    print("Perceptron")
    mainPerceptron()