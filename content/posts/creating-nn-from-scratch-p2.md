---
title: Creating a neural network from scratch in Python
date: '2018-09-05'
subtitle: Part II

---
## Recap
You now know about perceptrons and how they work. You've learnt about sigmoid neurons and various other activation functions that can help a neural network learn without causing too much trouble. You've come this far, so congratulations! In this post, we'll talk about how do the neural networks work, and how they actually learn to recognise stuff.

## How do Neural Networks work?
Now that we know the basic building blocks, let's dive into some more depth. I'll demonstrate a simple neural net that achieves 62% accuracy on a small dataset -  do note that this is only for demonstration purposes, neural networks are known to achieve as high as 99% accuracy and training on datasets that have nearly 5 million to 50 million samples is not uncommon.

Now, I can't stress this point enough - in any machine learning system, data is of the utmost importance. Data collection and preprocessing almost takes up 90% of the time in developing a particular model. [Kaggle](http://kaggle.com/) is a very good source of datasets from all over the world. For our model, we'll be using [this dataset of Biomechanical features of orthopedic patients](https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients/downloads/column_2C_weka.csv/1).

The next step is to determine the __network hyperparameters__. Just what do I mean by that? Well, hyperparameters are something that restrict your parameters. For instance, one hyperparameter would be how many nodes do you want in each layer of the network? To help you remember:

![A simple perceptron](/images/perceptron.png)


Each circle was a node and each connection had a weight associated with it. The number of circles in each layer is a hyperparameter that we can change. As the number of nodes (read : hyperparameter) changes, the number of weights also change, thereby changing the network as a whole.

If you have downloaded the dataset, you'll see that there are 6 features and 1 label. Obviously, the  number of inputs will be equal to the number of features and the number of outputs will be equal to the number of label(s). This is because you are essentially trying to find some dependency function that maps your inputs to their corresponding outputs. Makes sense?

Let us visualise our network. Personally, I believe that visualising a network can help you get a basic understanding of how the work is going on. We'll be using a three layered network having 7 nodes in the input layer, 10 nodes in the hidden and 1 node in the output layer, which will somewhat look like this.


![Our network](/images/network.png)


18 nodes, 80 edges in total.
Let's start implementing this architecture and understanding how this works. We'll begin by importing libraries like numpy and matplotlib. I'm assuming you have familiarity with these. If not, fear not, it's pretty simple - I'll explain as we go.

```
import numpy as np
import matplotlib.pyplot as plt
```
`matplotlib.pyplot` is used to plot the error graph, i.e. to see how well our model is learning.

Next, let's import the dataset.

```
dataset = np.genfromtxt('data/column_2C_weka.csv',skip_header=1,delimiter=',')
X = dataset[:,:-1]
Y = dataset[:,-1:]
```
 `skip_header`  skips the number of rows from the file given as argument.  The other parameters are pretty self-explanatory.

Now, `dataset[:,:-1]` means all rows (:) and all columns except the last one (:-1) would be loaded into `X`. We do this because the last column contains the labels. If it is not clear by now, `X` is the feature matrix, and `Y` is the label vector. Simple inputs and outputs, really.

Similarly, `dataset[:,-1:]` would load all rows and only the last column into `Y`.

One thing though. This dataset contained labels as strings, which is extremely inconvenient to process, given that the rest of the data are pure numbers. This can be easily resolved by simply opening up a text editor and searching + replacing all instances of a particular label with a specific number. In my case, I have replaced them with 0 and 1, since there are two distinct classes.

Next, we'll preprocess our dataset. Since its relatively clean and simple, we have to do minimal preprocessing.

```
X = (X - X.mean())/X.std()
bias = np.ones((X.shape[0],1),dtype=X.dtype)
X = np.concatenate((X,bias),axis=1)
```

The first line performs what we call **normalization**.  Let me explain what that is. You see, more often than not, the numeric data is in various ranges. For instance, one feature may be in the range of 0-1, whereas another may be in the range of 1000-2000. This results in poor training of the network, since the weights start to rely heavily on the features with more range, even when the features in smaller ranges are also equally important. This is because of our activation function, if you remember

$$\begin{eqnarray} 
  output = \frac{1}{1+\exp(-\sum_j w_j x_j-b)}.
\end{eqnarray}$$

Note that w<sub>j</sub> and x<sub>j</sub> are multiplied. Obviously, the greater x is, the greater would be the product, and hence more would be the impact. To resolve this, we subtract the mean of each feature from the feature itself (one feature at a time) and divide it by its standard deviation. This ensures that all the features are in one particular range, and helps in increasing the efficiency of the network.

The next line adds a bias layer to the network. This is generally done when the input features may not have a direct correlation with the output labels. It can also be added when there are direct correlations, there would not be any significant effect on the network's performance.

Next, let's initialize the network. We'll use a function for that.

```
def initialize():
    '''
    initialize the weight matrices
    Input:
       None
    Output:
        parameters : randomly initialized weight matrix dictionary
    '''
    
    W1 = np.random.randn(7,10)            # 7 x 10
    W2 = np.random.randn(10,1)            # 10 x 1

    
    parameters = {"W1": W1, "W2":W2}
    
    return parameters
```
The `random.randn(a,b)` function returns a numpy array of `a x b` dimensions initialized randomly.

The next part is where we'll pass the outputs through the activation function. Basically, what happens is this : Each output from one node is passed through its connections to every other node in the immediately next layer, the outputs being multiplied with the respective weights and then passed through the activation function. This part is called **`FeedForwarding`** (machine learning engineers like to use complicated terms for very simple things - I know) and we'll implement this next.

```
def feedForward(X,parameters):
    '''
    Computes the feed forward propagation of neural networks
    Input : 
        X : Feature Matrix 
        parameters : Dictionary containing weights and biases 
    Output :
        cache : dictionary containing activations
    '''
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    
    A1 = X                             # 1 x 7
    A2 = np.tanh(np.dot(A1,W1))        # 1 x 4
    A3 = np.tanh(np.dot(A2,W2))        # 1 x 1
    
    cache = {"A1":A1, "A2":A2, "A3":A3}
    
    return cache
```
`tanh` is our go-to activation function here (just one of the many that I used - you can use others, too). We're using a dictionary cache to hold the activations, we'll need them later.

I think by now you've got an idea of how the network is computing. Basically, you have the input features connecting to hidden layers and them connecting to the outputs. Each part of a feature holds some importance (or none) , the degree of the importance is determined by - congratulations, you're right! -  the weights of the connections.

>The ultimate aim is to optimize the weights in such a manner that the network learns to predict the output correctly.

### The master stroke : Backpropagation
The real question is - how do we that?! So far we've only been moving forward, so there's no question of optimization - its plain and simple calculations till now. Here's the way we optimize the weights :

Once we've reached the output layer, we see what was our output and compare it to the real output. Remember, neural networks use supervised learning, so you already have a knowledge of what the correct outputs should be. Then, we check the error in the output by subtracting our output from the real one. Let's say, you have a 4 node output layer. The real output should be 1-0-0-0, whereas you got 0.2-0.3-0.0-0.8 in the same order. This implies that you have to increase the weight in the connection to the first node by a huge amount, decrease the weight in the connection to the second node a little, not to change the weight for the third, and decrease the weight for the fourth connection a huge amount. This same thing is repeated for each layer (while moving backwards), and the amount by which the weights decrease are governed by a few equations.

This is achieved using a technique called `Gradient Descent`. Intuitively, you can think of this as if you're going downhill. Imagine you're standing on top of a cliff, and you need to get to the bottom. The way you do that is - you look around you and determine the direction you should move in to get to the bottom in the least time, i.e. the steepest slope you can move down in. We sometimes also use an optimizer to speed Gradient Descent up, such as `AdaBoost` or `Adam optimizer`. Intuitively, an optimizer tells you how big should be your each step while moving downhill, larger initially, smaller as you reach close to the ground.

Let's implement backpropagation now.


```
def backPropagate(X, Y, parameters, cache, alpha, batch_size=1):
    '''
    Performs back propagation in the network
    Input :
        X : Feature Matrix
        Y : Label Matrix
        parameters : dictionary containing the weight matrices
        cache : dictionary containing the activation matrices
        alpha : learning rate
        batch_size : The number of instances to be processed at a time
    Output :
        parameters : dictionary containing updated weight matrices
    '''
    W1, W2 = parameters["W1"], parameters["W2"]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    
    dA3 = A3 - Y
    
    dW3 = np.dot(A2.T,dA3)/batch_size
    
    dA2 = np.dot(dA3,W2.T)*(1.0 - np.square(A2))
    
    dW2 = np.dot(X.T, dA2)
    
    W2 -= alpha*dW3
    W1 -= alpha*dW2
    
    parameters["W1"] = W1
    parameters["W2"] = W2
    
    return parameters
```


`dA3`, `dW3`  - these are just variables that store the outputs of the equations. Read [this awesome article by Brian Dolhansky](http://briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4)  to get a clearer picture about backpropagation and the equations in play.

The next part is the one where we link everything together. Basically, the driver function of our model.
```
def train(X,Y,alpha=0.001,batch_size=1,num_iterations=1000):
    '''
    Trains our neural network with the help of the functions declared above.
    Input:
        X : feature matrix
        Y : label matrix
        alpha : learning rate
        batch_size : no. of instances to be processed at a time
        num_iterations : no. of times the model iterates to improve
    Output:
        history : a list containing errors for plotting
    '''
    
    np.random.shuffle(X)
    parameters = initialize()
    history = []
    
    for epoch in range(num_iterations):
    
        error = 0
        
        for instance in range(0,X.shape[0],batch_size):
            
            split_idx = min(instance+batch_size, X.shape[0])
            X_batch = X[:split_idx][:]
            Y_batch = Y[:split_idx][:]
            
            cache = feedForward(X_batch,parameters)
            a3 = cache["A3"]
            error = 0.5*np.sum((Y_batch-a3)**2)/batch_size
            
            parameters = backPropagate(X_batch, Y_batch, parameters, cache, alpha, batch_size=1)
            
        if epoch%100==0 and epoch:
            history.append(error)
            print("Epoch : %s , error : %s" %(epoch,error))
            
            
    return (history,parameters)
```

The `history[]` list contains the errors in each iteration run through the network. When we plot it, we'll see that the error goes down. Let's run the driver module.

```
train_test_split_idx = int(0.8*dataset.shape[0])
history,parameters = train(X[:train_test_split_idx],Y[:train_test_split_idx],num_iterations=1000)
```
We split the dataset into 2 parts in a 80:20 ratio to evaluate how well the model performs on unseen data. The 20% part is generally referred to as hold-out set or cross-validation set. One successful run of the above snippet outputs this :
```
Epoch : 100 , error : 15.185622160437514
Epoch : 200 , error : 14.436979136799541
Epoch : 300 , error : 13.854835338354611
Epoch : 400 , error : 13.623602526940928
Epoch : 500 , error : 13.505695410869142
Epoch : 600 , error : 13.385360723596113
Epoch : 700 , error : 13.29948046981653
Epoch : 800 , error : 13.236722174019114
Epoch : 900 , error : 13.181646652646013
```

Plotting this, we get :

`_ = plt.plot(range(9),history)`


![Loss](/images/graph1.png)


The x-axis is the number of iterations in 100s, and y-axis is the error. You can see the error decreasing with each iteration.

Also, do note that since the network is not that robust, the error during training might also shoot up awkwardly and then start decreasing again, somewhat like this :

![Alternative way the loss might behave](/images/graph2.png)


Next, we evaluate the model using the hold-out set.

```
cache = feedForward(X[train_test_split_idx:],parameters)
predictions = cache["A3"]
count = np.sum(np.argmax(predictions,axis=1) == np.argmax(Y[train_test_split_idx:],axis=1))
print('Accuracy: %.2f'%(float(count)/float(Y[train_test_split_idx].shape[0]))+'%')
```

We get the output:
`Accuracy: 62.00%`

That's it!

I'm hoping you learnt at least some part of how neural networks work, if not the entire part of it - even that would be a huge success for me.

Feel free to tweak the network hyperparameters to get even more accuracy, and let me know if you have any problems in the comments. If you liked it - don't forget to share it others! [Here is the full code](https://github.com/srdg/blog/blob/master/_code/nn-from-scratch.py) if you want to read it.
Most importantly, have fun!