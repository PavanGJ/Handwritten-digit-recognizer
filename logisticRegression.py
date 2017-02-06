import datpro

from _global import *
from params import REG,RATE,PREC,RATIO,OUT,MNIST_TRAIN,MNIST_TEST,USPS_FOLDER

class BatchGD(object):
    #Class to perform Batch Gradient Descent.
    def __init__(self):
        #Method to perform class initializations
        self.rate = RATE                                                        #Learning Rate
        self.reg = REG                                                          #Regularization Constant
        self.prec = PREC                                                        #Cost precision required

    def costFunction(self,X,t,w):
        #Method to compute the cost of the system with respect to the inputs, current weights and target values
        #Inputs:
        #           X       ->  Input feature matrix
        #           t       ->  Target matrix
        #           w       ->  Weights of the regression system
        #Outpus:
        #           J       ->  Cost of the regression system with the weights

        #Method initializations
        m = t.shape[0]                                                          #   m  ->  Size of sample
        #End of initializations

        h = self.output(np.dot(X,w))                                            #   h   ->  hypothesis
        t = t.transpose()
        #Calculating log values to be used for cost computations
        hp1 = h.copy()                                                          #hp1    ->  hypothesis part 1
        hp1[hp1==0] = 10**-10                                                   #To eliminate the possibility of a log(0) = infinity
        logh = np.log(hp1)                                                      #logh = log(h)
        hp2 = 1 - h                                                             #hp2    ->  hypothesis part 2
        hp2[hp2==0] = 10**-10                                                   #To eliminate the possibility of a log(0) = infinity
        lognh = np.log(hp2)                                                     #lognh = log(1-h)

        #Setting the infinity values to 0
        logh[np.abs(logh)>=np.inf] = 1E308                                      #log(h) with infinity values scaled to 1x10^308
        lognh[np.abs(lognh)>=np.inf] = 1E308                                    #log(1-h) with infinity values scaled to 1x10^308

        cost = -(np.dot(t,logh)+np.dot((1-t),lognh))                            #Cost matrix
        cost += self.reg * np.dot(w.transpose(),w)/(2)                          #Performing regularization
        J = np.sum(cost,axis=0)/m                                               #Computing average cost for each subsystem

        return J

    def gradient(self,X,t,w):
        #Method to compute the gradient. Used to update the weights.
        #Inputs:
        #           X       ->  Input feature matrix
        #           t       ->  Target matrix
        #           w       ->  Weights of the regression system
        #Outpus:
        #           grad    ->  Gradient

        #Method initializations
        m = t.shape[0]                                                          #   m  ->  Size of sample
        #End of initializations

        h = self.output(np.dot(X,w))                                            #   h   ->  hypothesis
        grad = np.dot(X.transpose(),(h-t))/m                                    #Computing the gradient for each subsystem
        grad += self.reg*w/m                                                    #Performing Regularization

        return grad

    def gradientDescent(self,X,t,w):
        #Method to perform Gradient Descent.
        #Inputs:
        #           X       ->  Input feature matrix
        #           t       ->  Target matrix
        #           w       ->  Initial weights of the regression system
        #Outpus:
        #           w       ->  Finalized weights

        #Entry point error checking
        #First check: to ensure that the target matrix is a two dimensional array
        try:
            (m,n) = t.shape
        except:
            print "DimensionError: target matrix dimensions are not congruent with the specification"
            exit(0)

        #Second check: to ensure that the feature matrix is a two dimensional array
        try:
            (M,N) = X.shape
        except:
            print "DimensionError: feature matrix dimensions are not congruent with the specification"
            exit(0)

        #Third check: to ensure that the weight matrix is a two dimensional array
        try:
            (wN,wn) = w.shape
        except:
            print "Dimension Error: weight matrix dimensions are not congruent with the specifications"
            exit(0)

        #Fourth check: to ensure the target matrix and feature matrix have the same number of samples
        if m != M:
            print "DimensionError: number of samples in feature matrix and target matrix vary"
            exit(0)

        #Fifth check: to ensure dimensions of the weight matrix is congruent with the input data
        if wN != N or wn != n:
            print "DimensionError: weight matrix dimensions are not congruent with the input data"
            exit(0)
        #End of error checking

        #Method initializations
        m = t.shape[0]                                                          #   m  ->  Size of sample
        c = t.shape[1]                                                          #   c  ->  Number of classes
        rate = np.ones((1,c))*self.rate                                         #   rate  ->  Learning Rates for all classes
        #End of initializations

        J = []                                                                  #An array to store the Costs
        init_cost = self.costFunction(X,t,w)                                    #Computing the initial Cost
        J.append(init_cost)


        loop = True                                                             #Loop flag
        while(loop):
            grad = self.gradient(X,t,w)                                         #Computing Gradient
            w = w - (rate * grad)                                               #Updating weights
            cost = self.costFunction(X,t,w)                                     #Computing the system cost with new weights
            J.append(cost)
            rate = ((np.abs(J[-1]-J[-2]))>=self.prec) * rate
            if ((np.abs(J[-1]-J[-2]))<=self.prec).all():                        #If desired precision achieved
                loop = False                                                    #Loop is set to end

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
            m = z.shape[0]                                                      #Storing number of samples for later reshaping
            e = np.exp(z)                                                       #Computing exponential values
            d = np.sum(e,axis=1).reshape((m,1))                                 #Calculating the sum for each input and reshaping as a row matrix
            return e/d
        else:
            return (1/(1+np.exp(-1*z)))

class Error(object):
    #Class to perform error checking validation
    def __init__(self):
        #Method to initialize the class variables
        pass

    def error(self,X,t,w):
        #Method to compute the error value
        #Inputs:
        #           X       ->  Input feature matrix
        #           t       ->  Target matrix
        #           w       ->  Initial weights of the regression system
        #Outpus:
        #           e       ->  Error value

        #Method initializations
        m = t.shape[0]                                                          #   m  ->  Size of sample
        #End of initializations

        h = self.output(np.dot(X,w))                                            #Computing the hypothesis value
        #Compare the predicted value with the known value and update
        #error count if not equivalent

        e = np.sum(np.argmax(h,axis=1)!=np.argmax(t,axis=1))/float(m)           #Error = Count of errors/Total Samples

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
            m = z.shape[0]                                                      #Storing number of samples for later reshaping
            e = np.exp(z)                                                       #Computing exponential values
            d = np.sum(e,axis=1).reshape((m,1))                                 #Calculating the sum for each input and reshaping as a row matrix
            return e/d
        else:
            return (1/(1+np.exp(-1*z)))

class LogReg(object):
    #Class to perform Logistic Regression
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

    def data_split(self,ratio):
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
            m = z.shape[0]                                                      #Storing number of samples for later reshaping
            e = np.exp(z)                                                       #Computing exponential values
            d = np.sum(e,axis=1).reshape((m,1))                                 #Calculating the sum for each input and reshaping as a row matrix
            return e/d
        else:
            return (1/(1+np.exp(-1*z)))

    def predict(self,X):
        #Method to predict the values
        #Inputs:
        #           X       ->  Input feature matrix
        #Outpus:
        #           h       ->  Predicted values

        #Error checking
        try:
            if X.shape[1] != self.w.shape[0] :                                  #Error checking to check if the required number of weight
                                                                                #have been provided
                print "Weights not in congruence with the system"
                exit(0)
        except:
            print "DimensionError: The input should be a 2D Matrix"
            exit(0)
        #End of error checking

        #Method initializations
        #End of initializations

        h = self.output(np.dot(X,self.w))                                       #Computing the hypothesis value

        return h

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
        [train,validate] = self.data_split(RATIO)                               #Split input data based on the ratio
        e = Error()                                                             #Initializing error class
        #End of initializations
        ########################################################################
        #Start of Training

        #Extracting Feature Matrices and Target values for training dataset
        [X_train,t_train] = train

        #One Hot Encoding of the target values
        t_train = self.oneHotEncode(t_train)

        print "Training the system. Please Wait"                                #Message for the user
        self.w = self.train(X_train,t_train)                                    #Train the system

        train_time = time()                                                     #Storing the end of training time stamp

        #End of training
        ########################################################################
        #Start of validation

        #Extracting Feature Matrices and Target values
        [X_validate,t_validate] = validate

        #One Hot Encoding of the target values
        t_validate = self.oneHotEncode(t_validate)

        error = e.error(X_validate,t_validate,self.w)                           #Compute error using validation set
        print "Validation Accuracy: ", 1-error
        #End of validation
        ########################################################################
        #Start of testing

        print "Testing the system.Please Wait"
        self.X_test = self.normalize(self.X_test)                               #normalize test dataset
        self.X_test = self.bias(self.X_test)                                    #Add bias unit to the test dataset

        #One Hot Encoding of the target values
        self.t_test = self.oneHotEncode(self.t_test)

        error = e.error(self.X_test,self.t_test,self.w)                         #Computing test error
        print "Test Accuracy: ", 1-error

        #End of testing
        ########################################################################
        #Start of Testing on USPS dataset

        print "Testing on USPS dataset"

        [self.X_test,self.t_test] = datpro.USPS(USPS_FOLDER)
        self.X_test = self.normalize(self.X_test)                               #normalize test dataset
        self.X_test = self.bias(self.X_test)                                    #Add bias unit to the test dataset

        #One Hot Encoding of the target values
        self.t_test = self.oneHotEncode(self.t_test)

        error = e.error(self.X_test,self.t_test,self.w)                         #Computing test error
        print "Test Accuracy: ", 1-error

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
        w = np.random.random([m*n]).reshape((m,n))

        #Initializing gradient descent algorithm based on the type mentioned
        gradDescent = BatchGD()

        w = gradDescent.gradientDescent(X,t,w)                                  #Performing gradientDescent

        return w
