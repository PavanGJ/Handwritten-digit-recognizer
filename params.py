#Global Parameters File
#Purpose:   To declare global parameters that need to be changed for each traning set

#PATHS TO FOLDERS
USPS_FOLDER = './USPS_data'
MNIST_TRAIN = './MNIST_data'
MNIST_TEST = './MNIST_data'

#Constants declaring Learning Rates
RATE = 1
NNRATE = 0.03

#Constants declaring Regularization Constants
REG = 1
NNREG = 0.0000

#Precision Values
PREC = 0.001

#Activation Function Type
#Possible values : 'RELU', 'IDENTITY', 'SOFTPLUS', 'TANH', 'SIGMOID'
ACT = 'SIGMOID'

#Output Function Type
#Possible values : 'SOFTMAX', 'SIGMOID'
OUT = 'SOFTMAX'


#Input Dataset Split Ratio
RATIO = 0.8

#Number of Hidden Layers
NHL = 1

#Hidden Layer Length
HLLENGTH = 900
