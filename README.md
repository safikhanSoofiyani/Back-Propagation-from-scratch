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
# Wandb Functionality:

1. To `use wandb mode`, find your `API key` from your wandb account and paste it in the output box after you executed this code snippet :     
  ```python
!wandb login --relogin
# enter the entity and project name in these variables
entity_name="_entity_name_"
project_name="_project_name_"
  ```
2. You can `perform experiments` by running the sweeps, using this function:
```python
sweeper(entity_name,project_name)
```
3. You can `compare` the performance of two `loss functions` by using this function:
```python
loss_compare_sweeper(entity_name,project_name)
```
4. You can plot the `confusion matrix` for the test dataset by using this function, this returns predicted labels and true labels:
```python
y_pred,y_t=plot_confmat_wandb(entity_name,project_name)
``` 
--------------------------------------------------------------     
# Available options to customize the Neural Network:
  
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
