# CS6910-Assignment1
<h2> Assignment 1 : Backpropagation. (CS6910 - Fundamentals of Deep Learning) <h3>
Authors :  Vamsi Sai Krishna Malineni(OE20S302) and Mohammed Safi Ur Rahman Khan(CS21M035) 

<h4>
  1) The "CS6910-Assignment1.ipynb " jupyter notebook represents our main submission. The code structure of the main notebook is such that it can be run sequentially till the end by pressing "Shift + Enter" for each cell, alternatively one can use "Run all cells"
 
<h4> 2) We are using a tool called "wandb.ai" which helps to keep track of large number experiments to find the best model. This tool can be accessed by the following code snippet :
  
  ```python
  !pip install wandb -qqq
  import wandb
  ```
 <h4> 3) The dataset used for training the model is obtained from "keras". The function "prepare_data()" splits the dataset into training,testing and validation data. This function returns the labels of the images as one-hot encoded vector.
<h4> Dataset can be downloaded using this code snippet:
  
  ```python
    from keras.datasets import fashion_mnist
  ```
   <h4> 4) The sample images of each class can be seen on the local machine by running this function: "plot_locally()"
    
   <h4> 6) In order to run a wandb sweep, the hyper_parameters have to be set up.
     <h4> Run the following code snippet to setup the hyper_parameters, you can change them according to your experiment's requirement :
     
     
  ```python
     hyperparameters = {
    "learning_rate":{
       'values': [0.001, 0.0001]
    },

    "number_hidden_layers": {
        'values' : [3, 4, 5]
    },

    "number_neurons": {
       'values': [32, 64, 128]
    },

    "initialization_type": {
        'values' : ["xavier", "random"]
    },

    "activation_function": {
        'values': ["sigmoid", "tanh", "relu"]
    },

    "mini_batch_size": {
        'values': [16,32,64,128]
    },

    "max_epochs": {
        'values': [5, 10, 20]
    },

    "lambd": {
        'values': [0, 0.0005, 0.5]
    },

    "optimization_function": {
        'values': [mini_batch_gd, momentum_gd, nesterov_gd, rmsprop, adam, nadam]
    }

}

  ```
     
   <h4> 7) To run a wandb sweep, use the following code snippet :
     
     
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
