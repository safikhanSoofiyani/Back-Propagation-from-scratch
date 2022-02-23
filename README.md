# Assignment 1 : Developing Backpropagation from scratch
-----------------------------------------------------------
In this assignment, we developed a feed forward neural network from scratch. We have used gradient descent method and its variants as optimization algorithm with backpropogation to classify images from Fashion-MNIST dataset. We used "wandb.ai" to perform experiments for hyperparameter tuning. 
# Libraries and their application :
1. Numpy: Mathematical operations are performed by this library
2. Keras: This library is used to obtain the dataset.
3. Matplotlib and Seaborn: Sample images from each class and Confusion Matrix are plotted using these libraries respetively
4. sklearn: The dataset is split into Train-Test-Validation by this library
5. wandb: This library is used to log the metrics to wandb.ai.
# Installations:
The above mentioned libraries can be installed on local machine by using the following code snippet in the command prompt:
```python
pip install numpy
pip install keras
pip install matplotlib
pip install seaborn
pip install sklearn
pip install wandb
```
If you are running the code on Google colab, all the above mentioned libraries are already installed **except** "wandb". Add the following code in a cell
```python
!pip install wandb
```
# Code Usage:
1. The entire code is modularised using functions. The code structure is such that it can be run sequentially till the end by pressing **SHIFT+ENTER** for each cell, alternatively you can use **Run all cells**
2. If you wish to get results of a specific hyperparameter configuration on the local machine
3. To run a wandb sweep, use the following code snippet :     
     
  ```python
     
  sweep_config = {
    'method' : 'bayes',
    'metric' :{
        'name': 'Validation_Accuracy',
        'goal': 'maximize'
    },
    'parameters': hyperparameters
  }
     
     sweep_id = wandb.sweep(sweep_config, entity="", project="")
     wandb.agent(sweep_id, train)
  ```
     
<h3> Available options to customize the Neural Network:
  
<h4> 1) Loss functions
  
  ```python
  
  MSE()
  CrossEntropy()
  ```
<h4> 2) Optimization functions
  
  ```python
  
  mini_batch_gd()
  momentum_gd()
  nesterov_gd()
  rmsprop()
  adam()
  nadam()
  ```
<h4> 3) Weight Initializations
  
  ```python
  Xavier()
  Random()
  ```
<h4> 4) Activation Functions
  
  ```python
  sigmoid()
  tanh()
  relu()
  softmax()
  ```
