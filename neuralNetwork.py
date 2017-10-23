import datpro

from _global import *
from params import NNRATE,NHL,HLLENGTH,RATIO,OUT,NNREG,MNIST_TRAIN,MNIST_TEST


class StochaisticGD(object):
    #Class to perform Stochaistic Gradient Descent.
    def __init__(self):
        #Method to perform class initializations
        self.rate = NNRATE                                                      #Learning Rate
        self.reg = NNREG                                                        #Regularization Constant

    def activation(self,z):
        #Method to compute the activation values
        #Inputs:
        #           z       ->  Input values
        #Outputs:
        #           a       ->  Activation values with bias appended

        a = 1/(1+np.exp(-1*z))                                                  #Sigmoid activation function
        a = np.concatenate((np.ones((1,1)),a))

        return a

    def backprop(self,A,w,t,y):
        #Method to perform back propogation to calculate the gradient for the weights
        #Inputs:
        #           A       ->  Activation matrix
        #           w       ->  weights of the system
        #           t       ->  Target values
        #           y       ->  predicted target values
        #Outputs:
        #           d       ->  Backpropagation differences

        m = len(w)

        #Backpropagation for output layer
        d = [y-t]
        #Backpropagation for all other layers
        for i in range(m):
            index = (m-1)-i
            z = np.dot(w[index].transpose(),d[-1])*self.derivative(A[index])
            d.append(z[1:,:])

        diff = []
        for i in range(len(d)):
            diff.append(d.pop())

        return diff

    def derivative(self,a):
        #Method to compute the derivative of output activation values
        #Inputs:
        #           a       ->  Activation values
        #Outputs:
        #           d       ->  derivative of a

        d = a*(1-a)

        return d

    def feedforward(self,X,w):
        #Method to perform feed forward propogation as an intermediate step in calculation of gradients for the weights
        #Inputs:
        #           X       ->  Input feature
        #           w       ->  Weights of the regression system
        #Outputs:
        #           y       ->  Predicted values
        #           A       ->  Activation values for all the hidden layers

        m = len(w)
        A = [X]

        #Feed forward for all layers except the output layer
        for i in range(m-1):
            z = np.dot(w[i],A[-1])
            A.append(self.activation(z))

        #Feed forward propogation for output layer
        z = np.dot(w[i+1],A[-1])
        y = self.output(z)

        return [y,A]


    def gradient(self,X,t,w):
        #Method to compute the gradient. Used to update the weights.
        #Inputs:
        #           X       ->  Input feature
        #           t       ->  Target matrix
        #           w       ->  Weights of the regression system
        #Outputs:
        #           grad    ->  Gradient

        m = len(w)
        grad = []                                                               #Gradients
        [y,A] = self.feedforward(X,w)
        d = self.backprop(A,w,t,y)

        for i in range(m):
            D = np.dot(d[i+1],A[i].transpose())
            D = D + self.reg*w[i]
            grad.append(D)

        return grad

    def gradientDescent(self,X,T,w):
        #Method to perform Gradient Descent.
        #Inputs:
        #           X       ->  Input feature matrix
        #           T       ->  Target matrix
        #           w       ->  Initial weights of the regression system
        #Outpus:
        #           No outputs
        m = X.shape[0]
        M = X.shape[1]
        N = T.shape[1]
        for i in range(m):
            x = X[i].reshape((M,1))
            t = T[i].reshape((N,1))
            grad = self.gradient(x,t,w)
            g = len(grad)
            for j in range(g):
                w[j] -= self.rate*grad[j]


        return w

    def output(self,z):
        #Method to compute the output value of given input
        #Input:
        #           z       ->  Input value or matrix
        #Output:
        #           g       ->  output value or matrix for the input.

        #Softmax:
        #           f(x) = e^x_k/SUM(e^x_j)    :: for j in K
        #Sigmoid:
        #           f(x) = 1/(1+e^-x)
        if OUT == 'SOFTMAX':
            e = np.exp(z)                                                       #Computing exponential values
            d = np.sum(e)                                                       #Calculating the sum of all exponentials

            return e/d
        else:
            return (1/(1+np.exp(-1*z)))

class Error(object):
    #Class to perform error checking validation
    def __init__(self):
        #Method to initialize the class variables
        pass

    def activation(self,z):
        #Method to compute the activation values
        #Inputs:
        #           z       ->  Input values
        #Outputs:
        #           a       ->  Activation values with bias appended

        a = 1/(1+np.exp(-1*z))                                                  #Sigmoid activation function
        a = np.concatenate((np.ones((1,1)),a))

        return a

    def error(self,X,T,w):
        #Method to compute the error value
        #Inputs:
        #           X       ->  Input feature matrix
        #           T       ->  Target matrix
        #           w       ->  Weights of the regression system
        #Outpus:
        #           e       ->  Error value

        #Error checking
        if w.shape[0] != NHL+1:                                                 #Error checking to check if the required number of weight
                                                                                #have been provided
            print "Weights not in congruence with the system"
            exit(0)
        #End of error checking

        #Method initializations
        m = len(w)
        e = 0
        n = len(X)
        for i in range(n):
            A = [X[i].reshape((len(X[i]),1))]
            t = T[i].reshape((len(T[i]),1))
            for j in range(m-1):
                z = np.dot(w[j],A[-1])
                A.append(self.activation(z))

            #Feed forward propogation for output layer
            z = np.dot(w[j+1],A[-1])
            h = self.output(z)

            e += (np.argmax(h)!=np.argmax(t))

        e = e/float(n)                                                         #Error = Count of errors/Total Samples

        return e

    def output(self,z):
        #Method to compute the output value of given input
        #Input:
        #           z       ->  Input value or matrix
        #Output:
        #           g       ->  output value or matrix for the input.

        #Softmax:
        #           f(x) = e^x_k/SUM(e^x_j)    :: for j in K
        #Sigmoid:
        #           f(x) = 1/(1+e^-x)
        if OUT == 'SOFTMAX':
            e = np.exp(z)                                                       #Computing exponential values
            d = np.sum(e)                                                       #Calculating the sum of all exponentials
            return e/d
        else:
            return (1/(1+np.exp(-1*z)))

class NeuralNetwork(object):
    #Class to perform classification based on using Neural Networks
    def __init__(self):
        #Method to initialize the class variables
        #Initializing the dataset
        [self.X_train,self.t_train] = datpro.Training(MNIST_TRAIN)
        [self.X_test,self.t_test] = datpro.Testing(MNIST_TEST)

    def bias(self,X):
        #Method to add bias to the input data.
        #Inputs:
        #       X       ->  Input Feature Matrix (M x N)
        #Outputs:
        #       X       ->  Input Feature Matrix with an additional bias value (M x N+1)

        try:
            (M,N) = X.shape
        except:
            print "DimensionError: Input Feature Matrix dimensions not as specified"
            exit(0)

        X = np.concatenate((np.ones((M,1)),X),axis=1)                           #Concatenating a bias value

        return X

    def dataSplit(self,ratio):
        #Method to split data into train and validation sets
        #Inputs:
        #       ratio   ->  Split ratio
        #Outpus:
        #       [train,validate]    ->  Train set and Validation set

        #Method initializations
        m = self.t_train.shape[0]
        #End of initializations

        end = int(np.round(ratio*m))                                            #Computing the split point

        #Splitting data to train and validation sets
        train = [self.X_train[:end,:],self.t_train[:end,:]]                     #Train Dataset
        validate = [self.X_train[end:,:],self.t_train[end:,:]]                  #Validation Dataset

        return [train,validate]

    def normalize(self,X):
        #Method to normalize the input data
        #Inputs:
        #       X   ->  Input Feature Matrix
        #Outputs:
        #       X   ->  Normalized Input Feature Matrix
        #Normalization technique: Min-Max Normalization
        #           (value - min) /(max - min)

        Xmin = np.min(X)                                                        #Calculating Min value
        Xmax = np.max(X)                                                        #Calculating Max value
        X = (X - Xmin)/float(Xmax - Xmin)                                       #Feature Scaling/ Normalizing input matrix

        return X

    def oneHotEncode(self,t):
        #Method to perform One Hot Encoding over the given input
        #Inputs:
        #       t       ->  Input row matrix
        #Outputs:
        #       t       ->  One Hot Encoded values
        #Error Checking
        try:
            if t.shape[1]!=1:
                print "DimensionError:  should be a row matrix"
                exit(0)
        except:
            print "DimensionError:  should be a row matrix"
            exit(0)
        #End of error checking

        #Method initializations
        m = t.shape[0]                                                          #   m  ->  Size of sample
        values = np.unique(t)                                                   #Unique values
        n = len(values)                                                         #   n  ->  Number of classes
        #End of initializations

        #Logic:
        #Create a m*n matrix of 1's and then broadcast the unique
        #classification values across the rows
        #Example:
        #       unique classes -> [1 3 6 8]
        #       m -> 2      n -> 4
        #Creating a 2x4 matrix of 1's
        #           [1 1 1 1]
        #           [1 1 1 1]
        #Broadcasting and multiplying the values of unique classes across rows
        #           [1 3 6 8]
        #           [1 3 6 8]
        #Compare target values and if equal set value to 1 else 0
        #       t -> [1 6] {Transpose}
        #Output:
        #           [1 0 0 0]
        #           [0 0 1 0]

        temp = np.ones((m,n))                                                   #Creating a mxn matrix of 1's
        temp = temp*values                                                      #Broadcasting unique values across rows
        t = np.array((temp == t), dtype=np.uint8)                               #Comparing values

        self.classes = n                                                        #Storing the number of classes for use during prediction
        return t

    def output(self,z):
        #Method to compute the output value of given input
        #Input:
        #           z       ->  Input value or matrix
        #Output:
        #           g       ->  output value or matrix for the input.

        #Softmax:
        #           f(x) = e^x_k/SUM(e^x_j)    :: for j in K
        #Sigmoid:
        #           f(x) = 1/(1+e^-x)
        if OUT == 'SOFTMAX':
            e = np.exp(z)                                                       #Computing exponential values
            d = np.sum(e)                                                       #Calculating the sum of all exponentials
            return e/d
        else:
            return (1/(1+np.exp(-1*z)))

    def predict(self,X):
        #Method to predict the values
        #Inputs:
        #           X       ->  Input feature matrix
        #Outpus:
        #           e       ->  Error value

        #Error checking
        try:
            if X.shape[1] != self.w[0].shape[0] :                               #Error checking to check if the required number of weight
                                                                                #have been provided
                print "Weights not in congruence with the system"
                exit(0)
        except:
            print "DimensionError: The input should be a 2D Matrix"
            exit(0)
        #End of error checking

        #Method initializations
        m = len(w)                                                              #Number of layers
        n = len(X)                                                              #Number of inputs
        y = np.empty((0,self.classes))
        for i in range(n):
            A = [X[i].reshape((len(X[i]),1))]
            t = T[i].reshape((len(T[i]),1))
            for j in range(m-1):
                z = np.dot(self.w[j],A[-1])
                A.append(self.activation(z))

            #Feed forward propogation for output layer
            z = np.dot(self.w[j+1],A[-1])
            h = self.output(z)
            y = np.concatenate((y,h),axis=0)

            return y

    def run(self):
        #Method to execute the Logistic Regression
        #Inputs:
        #           None
        #Outputs:
        #           None

        #Method initializations
        start_time = time()                                                     #Storing the system time stamp
        self.X_train = self.normalize(self.X_train)                             #Feature Scaling input values
        self.X_train = self.bias(self.X_train)                                  #Adding a bias value
        [train,validate] = self.dataSplit(RATIO)                                #Split input data based on the ratio
        e = Error()                                                             #Initializing error class
        #End of initializations
        ########################################################################
        #Start of Training

        #Extracting Feature Matrices and Target values for training dataset
        [X_train,t_train] = train

        #One Hot Encoding of the target values
        t_train = self.oneHotEncode(t_train)

        self.w = self.train(X_train,t_train)                                    #Train the system

        train_time = time()                                                     #Storing the end of training time stamp

        #End of training
        ########################################################################
        #Start of validation

        #Extracting Feature Matrices and Target values
        [X_validate,t_validate] = validate

        #One Hot Encoding of the target values
        t_validate = self.oneHotEncode(t_validate)

        error = e.error(X_validate,t_validate,self.w)                                #Compute error using validation set
        print "Accuracy: ", 1-error
        #End of validation
        ########################################################################
        #Start of testing

        self.X_test = self.normalize(self.X_test)                               #normalize test dataset
        self.X_test = self.bias(self.X_test)                                    #Add bias unit to the test dataset

        #One Hot Encoding of the target values
        self.t_test = self.oneHotEncode(self.t_test)

        error = e.error(self.X_test,self.t_test,self.w)                              #Computing test error
        print "Test Accuracy: ", 1-error

        #End of testing
        ########################################################################
        end_time = time()
        print "Training Time: %ss"%((train_time-start_time))
        print "Total Time: %ss"%((end_time-start_time))


    def train(self,X,t):
        #Method to train the Logistic regression
        #Inputs:
        #       X       ->  Input feature matrix
        #       t       ->  Target values
        #Outputs:
        #       w       ->  trained weights
        #Extracting dimensions of the weight matrix
        try:
            m = X.shape[1]
            n = t.shape[1]
        except:
            print "DimensionError: dimensions of inputs not as specified"
            exit(0)

        #Initializing the weight matrix
        w = self.weights_init(m,n)

        #Initializing gradient descent algorithm based on the type mentioned
        gradDescent = StochaisticGD()

        w = gradDescent.gradientDescent(X,t,w)                                  #Performing gradientDescent

        return w

    def weights_init(self,m,n):
        #Method to initialize the weights of the system
        #Inputs:
        #           m       ->  Input layer Length
        #           n       ->  Output layer length
        #Outputs:
        #           w       ->  random weights for the system

        #Weights initialized by mapping one layer to the immediate next layer
        #Weight matrix size determined by the number of units/nodes in the layer
        #and the number of units/nodes in the next layer
        #If 'a' is the number of nodes in the layer and 'b' is the number of nodes in the next
        #layer, then weight matrix dimensions are:
        #               w_i   ->    b x a+1

        #Method initializations
        w = []

        #Input layer weights
        w_i = np.random.random((HLLENGTH,m))*0.01                                    #Initializing weights according to logic mentioned
        w.append(w_i)
        for i in range(NHL-1):                                                  #For all hidden layer except the last
            w_i = np.random.random((HLLENGTH,HLLENGTH+1))*0.01
            w.append(w_i)

        #Final hidden layer weights
        w_i = np.random.random((n,HLLENGTH+1))*0.01
        w.append(w_i)

        return np.array(w)
