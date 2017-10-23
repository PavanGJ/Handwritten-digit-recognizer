# Handwritten digit recognition system.

This project was implemented as an assignment for the course _**CSE574 : Introduction to Machine Learning**_ at _University at Buffalo, The State University of New York_ in Fall 2016. The goal of the project is to develop and compare different Machine Learning systems to recognise and classify handwritten digits.

### Dataset ###

The system developed was trained on [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The dataset consists of 70,000 handwritten digit image samples and is partitioned to two sets containing 60,000 samples and 10,000 samples respectively. The larger of the sets is used for training and the smaller is utilized for testing. 

The training set consisting of 60,000 images is further partitioned during training to form the validation set. To this end, 20% of the data i.e. 12,000 image samples have been repurposed to form the validation set.

### Implementation ###

The following machine learning systems have been developed.
* <b>Softmax Logistic Regression</b>
> A softmax regression system was developed and trained on the dataset. The system utilized <b>Gradiend Descent</b> algorithm for weight optimizations.
* <b>Single-Layered Neural Network</b>
> A single-layered neural network system was developed and trained on the dataset. The network utilized <b>Stochastic Gradient Descent</b> algorithm for weight optimizations and used <b>Sigmoid function</b> for hidden layer activation and a softmax regression output layer.

Both the systems utilize Cross Entropy Error function as the loss function

### Hyperparameter Tuning ###

Finding the optimal results for the system involved tuning hyperparameters to the optimal values. The following hyperparameters were tuned to the optimal values.
* L2 regularization constant - <b><i>&lambda;</i></b>
* Learning rate - <b><i>&eta;</i></b>
* \# of units in hidden layer

The hyperparameters of the system have been tuned by iteratively running the system over different values.

### Results ###

The results obtained are as follows:
* <b>Softmax Logistic Regression</b>

    Dataset     | Accuracy
    :------------:|:------------------:
    **Validation**  | 91.6%
    **Test**        | 91.67%

* <b>Single-Layered Neural Network</b>

    Dataset     | Accuracy
    :------------:|:------------------:
    **Validation**  | 93.26%
    **Test**        | 92.98%

* Additionally a **Convolutional Neural Network** was built using _**Tensorflow**_ similar to the system defined on the [_Tensorflow_ Website](https://www.tensorflow.org/get_started/mnist/pros)
The system was run for 2000 iterations instead of 20,000 iterations and a test accuracy of 97.59% was achieved on the dataset.
