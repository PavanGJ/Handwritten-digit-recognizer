#Global imports
from _global import np
import struct, os

class Training(object):
    #Class to format the training inputs
    #Usage :
    #       [X,t] = Training(path)
    #
    def __init__(self,path):
        self.dataset = []                                                       #Final output list
        iPath = path+'/train-images.idx3-ubyte'
        lPath = path+'/train-labels.idx1-ubyte'
        with open(lPath,'rb') as fLabel:
            magic, N= struct.unpack(">II", fLabel.read(8))
            labels = np.fromfile(fLabel,dtype=np.uint8)
        with open(iPath,'rb') as fImage:
            magic, num, rows, cols = struct.unpack(">IIII", fImage.read(16))
            images = np.fromfile(fImage,dtype=np.uint8).reshape((N, rows, cols))
        images = images.reshape((N,rows*cols))
        labels = labels.reshape((N,1))
        self.dataset.append(images)
        self.dataset.append(labels)

    def __repr__(self):
        return repr(self.dataset)

    def __getitem__(self,index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class Testing(object):
    #Class to format the training inputs
    #Usage :
    #       [X,t] = Training(path)
    #
    def __init__(self,path):
        self.dataset = []                                                       #Final output list
        iPath = path+'/t10k-images.idx3-ubyte'
        lPath = path+'/t10k-labels.idx1-ubyte'
        with open(lPath,'rb') as fLabel:
            magic, N= struct.unpack(">II", fLabel.read(8))
            labels = np.fromfile(fLabel,dtype=np.uint8)
        with open(iPath,'rb') as fImage:
            magic, num, rows, cols = struct.unpack(">IIII", fImage.read(16))
            images = np.fromfile(fImage,dtype=np.uint8).reshape((N, rows, cols))
        images = images.reshape((N,rows*cols))
        labels = labels.reshape((N,1))
        self.dataset.append(images)
        self.dataset.append(labels)

    def __repr__(self):
        return repr(self.dataset)

    def __getitem__(self,index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)