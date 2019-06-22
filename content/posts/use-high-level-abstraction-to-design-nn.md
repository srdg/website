---
layout: post
published: true
title: Using high level abstraction to design neural networks
date: '2018-09-21'
subtitle: Move faster from idea to code
tags:
 - deep learning
 - neural networks
categories:
 - deep learning 
---
If you have followed the last two posts on this blog, you'd know that we have implemented a neural network from scratch in python. However, the code that comes along with it might be difficult to understand for the uninitiated, because you need to have a basic understanding of linear algebra and partial derivatives as a basic prerequisite. Along with that we'd have to reuse most of the functions implemented there- for instance `feedforwarding` and `backpropagation` in every single model that we design. These might be a bit hard to grasp for the new users -- who have heard about this field recently and have a lot of creative ideas. As everyday the bar is set higher, we need more hands on board to unlock the potential that deep learning brings to the table, which is why libraries offering high level abstraction to the basic building blocks were designed. This was done so that new users can go from `idea phase` to `code phase` quickly, like illustrated in the following demographic.  

![The iterative process of ML](/images/machine-learning-iterative-process.png)  


Once you come up with an idea, these libraries give you the chance of quickly converting them into code and see what the results are. If they are not satisfactory enough, you again have the opportunity to change the hyperparameters (the experiment phase) and see the results without having to worry about the backend dependencies. One such library is `keras` and we'll see how to work with it today.  

Now, each of you can have a different system and various unresolved dependencies. To avoid conflicts, we'd use some open source tools among which one is [Google colab](https://colab.research.google.com), which is basically a Jupyter notebook available for you to use online. You also get free GPU services - 1xTesla K80 , compute 3.7, having 2496 CUDA cores and 12GB GDDR5 VRAM, should you choose to turn it on.  

But what shall we be working on? Well, like I said in my previous posts, Kaggle is a very good source of datasets, and one such recently uploaded dataset was [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn/home). A bit of background - for any company, its turnover depends a lot on its customers. Churning refers to the act of customers leaving/stopping to use the company's products, because of dissatisfatory outcomes/better options. Retaining older customers is much more cost effective than acquiring a new customer base (because for new customers you'd have to present a demonstration, pitch your sales and work your way up to earn their trust, while old customers already trust you and can vouch for your product), which is why churn rate is very important to the companies. We'll be analysing the dataset, and see the accuracy with which we can predict whether a given customer will churn or not.

## Set up the environment
First, we need to set up the execution environment. Even before that, we need to take care of a few things. 
### Enable the GPU
Go to [Google colab](https://colab.research.google.com) and open a new Python3 notebook.
To turn the GPU on, go to `Edit > Notebook Settings > Hardware accelerator` and select `GPU` from the dropdown menu.
### Download kaggle.json to use kaggle API
This is required so that you don't have to upload the dataset everytime you want to use it in colab. All you have to do is call the Kaggle API and it will manually download the dataset to the current space allotted to you by colab. In order to download `kaggle.json` you need to have a Kaggle account first. Once you are logged in, go to `My account > API > Create new API token` and click on it to download the JSON file.
Once you have downloaded it, we are all set to go. To set up the environment in colab,we'll use the following code:  
```
!pip install kaggle&>./kaggle.txt
!pip install keras&>./keras.txt
from google.colab import files
file = files.upload()
```
Once the last command executes, you'll get a prompt to upload a file. Upload the `kaggle.json` you just downloaded.  
That done, the next steps are required to use the kaggle API.  
```
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
```

That done, you need to copy the API command from the data tab [here](https://www.kaggle.com/blastchar/telco-customer-churn). The API will download a zip archive to the current working directory, and you need to unzip it to start working. This is done using -  
```
!kaggle datasets download -d blastchar/telco-customer-churn
!unzip *.zip
```
## Analyse the dataset
Next, we have to import the libraries we'll use.
```
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
```

We'll read the CSV file and display the first 5 records to get an idea of what we're dealing with.
```
df = pd.read_csv("./WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head(5)
```

`21 columns` shows quite a lot of features are available, so let us check out the unique value for each feature or attribute.

```
for attr in df.columns.values.tolist()[1:]:
  print("Unique values in "+attr+":")
  print(np.unique(df[attr]).tolist())
```
The `tolist()` function converts a `numpy.ndarray` type object to a `list` type object.

`df.columns.values.tolist()` would return a list of all attributes/features in the dataset. We ignored the first feature of `customerID` since it would be unique for every customer. After displaying the unique values, we come to realise that there are far too many unique values for `MonthlyCharges` and `TotalCharges`, so ignoring them, let us plot histograms for the rest of the attributes to get an idea of the variance each feature has.  


```
%matplotlib inline
columns = df.columns.values.tolist()[1:]
for attr in columns[:-1]:
  if attr in ('MonthlyCharges','TotalCharges'):
    continue
  _=plt.hist(df[attr].values,rwidth=0.85)
  plt.title(attr.upper())
  plt.xticks(rotation=90)
  plt.show()
```



`df.shape` would give us `(7043, 21)` - the shape of the dataset we are working with.



Next, its time to do some data preprocessing and cleaning.


```
df = df.applymap(lambda x: 1 if x == True or x in ('Yes','Male','DSL','Month-to-month','Bank transfer (automatic)') else x)
df = df.applymap(lambda x: 0 if x == False or x in ('No','Female','One year', 'Electronic check' ) else x)
df = df.applymap(lambda x: 2 if x in ('No phone service','Fiber optic','No internet service','Two year','Mailed check') else x)
df = df.applymap(lambda x: 3 if x=='Credit card (automatic)' else x)
df['TotalCharges']=df['TotalCharges'].apply(lambda x: 0 if x is None or x==' ' else float(x))
df = df.drop('customerID', 1)
df = df.fillna(0)
df.head(5)
x = df.as_matrix()
print(x.shape)
```  

Don't worry about the commands if you don't get them, here is a basic summary of what they are doing : 
all occurences of `Yes/Male/DSL/Month-to-Month/Bank transfer(automatic)` are converted to 1, all occurences of `No/Female/One year/Electronic check` are changed to 0 and all occurences of `No phone service/Fiber optic/No internet service/Two year/Mailed check` are converted to 2. All occurences of the only remaning unique value, `Credit card (automatic)` are converted to 3. Next, we clean the data by filling `NaN` and missing values/spaces with 0 and drop the feature of `customerID` and check out the first 5 columns to ensure the dataset was modified. The `df.as_matrix()` function converts our dataset into a `numpy.ndarray` type matrix with the same shape as that of original dataset passed to it. Since we have already dropped one feature (`customerID`), `x.shape` returns (7043, 20).

Next, we have to split this dataset into train and test sets and separate them into features(X) and labels(Y). We will then split the train set further into training and hold-out set for cross-validation, i.e. to ensure that our model does not go awry on seeing unknown samples.
We do this in the following code -- 

```
split_idx = int(0.8*7043)
X_train = x[:split_idx,:-1]
X_test = x[split_idx:,:-1]
Y_train = x[:split_idx,-1:]
Y_test = x[split_idx:,-1:]

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
```

Alternatively, you can also use the following code -- 

```
!pip install sklearn&>./sklearn.txt
y = x[:,-1:]
x = x[:,:-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=0.2, shuffle=True)
```
Both of them work the same. `print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)` 
will give you  
`(5634, 19) (5634, 1) (1409, 19) (1409, 1)`  
the shape of training and test sets.

## Designing the model
Now that we have the dataset ready to be fed into the model, we'll start with designing our neural network model. Here is where the abstraction comes in handy, you don't have to implement anything from scratch by yourself, a simple function call with all the right parameters would work. This could be done easily by any person with a basic understanding of the parameters and Python programming syntax.  
We'll start with importing the library and design the model.
```
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(19, input_dim=19, activation='relu'))
model.add(Dense(30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```
`Sequential` means the layers of nodes are going to be stacked one after another in sequence. The syntax is pretty simple -- the network we are using is a `19 - 30 - 8 - 1` node network, fully connected with `ReLu` as the activation function for the first three layers and `sigmoid` as the activation function for the last output layer. The first layer has 19 nodes because that is the dimension of our input vector. We are using `binary crossentropy` as the loss function , which can be stated as:  


$$L = -{(y\log(p) + (1 - y)\log(1 - p))}$$  


We are using `Adam` optimiser to optimise Gradient Descent, and accuracy of classification as the metric to determine the performace of our model. `Dense` refers to a fully connected layer, meaning every node of a layer is connected to every node of its preceding and following layer if they are present.
`model.compile()` creates the model ready to use for us. `model.summary()` returns a summary of the model, like this --
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_13 (Dense)             (None, 19)                380       
_________________________________________________________________
dense_14 (Dense)             (None, 30)                600       
_________________________________________________________________
dense_15 (Dense)             (None, 8)                 248       
_________________________________________________________________
dense_16 (Dense)             (None, 1)                 9         
=================================================================
Total params: 1,237
Trainable params: 1,237
Non-trainable params: 0
_________________________________________________________________
```

## Training and evaluation
Now that we have got the model ready to use, it is time to feed the data in it and see how it performs. But before feeding it, we have to take care of something else. Remember we had to split the training set further to get a training and a hold-out set for cross-validation, that ensures our model does not behave awkwardly on seeing unseen samples? Let us split the training set in a 90:10 ratio to create the training and hold-out sets.
```
%matplotlib inline
dev_idx=int(0.9*5634)
num_epochs = 40
obj = model.fit(X_train[:dev_idx,:], Y_train[:dev_idx,:], 
                epochs=num_epochs,
                batch_size=10,  
                verbose=1,
                validation_data=(X_train[dev_idx:], Y_train[dev_idx:]))
_=plt.plot(range(num_epochs),obj.history['acc'],'r-',range(num_epochs),obj.history['val_acc'],'g-')
plt.title('Accuracy graph')
plt.show()
_=plt.plot(range(num_epochs),obj.history['loss'],'r-',range(num_epochs),obj.history['val_loss'],'g-')
plt.title('Loss graph')
# round predictions
score = model.evaluate(x=X_test, y=Y_test)
print('Loss:', score[0])
print('Accuracy:', score[1])
```  
I am kind of a lazy person, so I didn't bother storing the training and hold_out sets in different variables. Instead, I identified the index at which the dataset should be split and sliced the matrix before feeding it to the model.  
This is what the training accuracy curve looks like. The green curve is for the training set, the red one is for the hold-out set.


![Accuracy curves](/images/trainacc.png)  
And this is for the loss calculated for each of 40 epochs.  
![Loss curves](/images/trainloss'.png)

In my case, the result came out to be -   
```
Loss: 0.429139761234362
Accuracy: 0.8041163946061036
```
Over 80% accuracy for unseen samples. Pretty much efficient, I'd say.  

## Quick recap!
+ Create a kaggle account and a google account (if you don't already have one).
+ Download `kaggle.json` from [kaggle](https://kaggle.com) and open a new Python3 notebook in [Google colab](https://colab.research.google.com).
+ Turn on the GPU accelerator in notebook.
+ Set up the execution environment by installing the libraries `keras` and `kaggle`.
+ Upload the `kaggle.json` file and change its location and permissions for kaggle API to work smoothly.
+ Download the dataset by using the API command and unzip it.
+ Import the libraries like `numpy`,`matplotlib`,`pandas`, `keras` etc.
+ Read the dataset and check out unique values for each feature.
+ Use some data visualisation techniques to determine the distribution of data and its variance (in this case, histogram).
+ Preprocess and clean the data to get rid of missing/blank/spaces/`NaN` values and encode the boolean or multiclass values.
+ Split the dataset into train, hold-out and test sets. Shuffle if the data is in an ordered fashion.
+ Design and compile the model with `keras`.
+ Feed the data into the model to train.
+ Plot the training curve and hold-out curve to determine the general trends for accuracy and loss. Preferably in different colours for training and cross-validation (in this case, red and green).
+ Evaluate the performance of the model on test set.

Here is the [full code](https://colab.research.google.com/drive/1DOGsMQL7YPt_OwIwvHOukQdD75ArJhMS) if you want to have a look. If you have any doubts or suggestions, do comment below!