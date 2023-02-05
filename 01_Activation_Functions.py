#!/usr/bin/env python
# coding: utf-8

# # Physics 494/594
# ## Activation Functions
# 

# In[2]:


# %load ./include/header.py
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import trange,tqdm
sys.path.append('./include')
import ml4s
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.style.use('./include/notebook.mplstyle')
np.set_printoptions(linewidth=120)
ml4s.set_css_style('./include/bootstrap.css')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ## Learning Goals
# 
# - Learn how to create simple functions in python
# - Understand that the output of a neuron is a non-linear *activation* function
# - Introduction to using built-in functions in `numpy` and simple plotting with `matplotlib`

# ## Perceptron

# In[3]:


def perceptron(z):
    if z <= 0: 
        return 0
    else:
        return 1
    
# this line allows us to easily plot the function for different values of z
perceptron = np.vectorize(perceptron)

# the different values of z we will consider
z = np.linspace(-10,10,100)


# In[4]:


plt.plot(z,perceptron(z))
plt.xlabel('z')
plt.ylabel('$a(z)$')


# ## Sigmoid
# 
# **Note:** We can get greek letters in the jupyter notebook by using `LaTeX` commands`+ TAB`, i.e. to get $\sigma$ you should type `\sigma + TAB`. 
# 
# Here we also introduce a `numpy` built-in function, the exponential `np.exp(...)`. 
# 
# We don't need to use `np.vectorize(...)` here as the exponential function is already ready to work with arrays.

# In[5]:


def σ(z):
    return 1.0/(1.0 + np.exp(-z))

plt.plot(z,σ(z))
plt.xlabel('z')
plt.ylabel('$a(z)$')


# ## Hyperbolic Tangent
# 
# Most trig (and hyperbolic trig) functions are also available in `numpy`

# In[6]:


plt.plot(z,np.tanh(z))
plt.xlabel('z')
plt.ylabel('$a(z)$')


# ## Rectified Linear Unit (ReLU)
# 
# Here we use the built-in `np.maximum(...)` function.

# In[7]:


def ReLU(z):
    return np.maximum(z, 0)

plt.plot(z,ReLU(z))
plt.xlabel('z')
plt.ylabel('$a(z)$')


# ## Leaky Rectified Linear Unit (Leaky ReLU)

# In[8]:


def leaky_ReLU(z):
    α = 1.0/10    
    return np.maximum(α*z, z)

plt.plot(z,leaky_ReLU(z))
plt.xlabel('z')
plt.ylabel('$a(z)$')


# <div class="span alert alert-success">
# <h2> Excercise </h2>
# Write a function that computes and plots the Exponential Leaky Unit defined by the piecewise continuous function: 
#     
# \begin{equation}
#     \mathrm{ELU}(z) =
#     \begin{cases}
#     \mathrm{e}^{z}-1 &;& z \le 0 \\
#     z &;& z >0
#     \end{cases}
# \end{equation}
# </div>
# 
# 

# In[9]:


# %load ./include/header.py
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import trange,tqdm
sys.path.append('./include')
import ml4s
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.style.use('./include/notebook.mplstyle')
np.set_printoptions(linewidth=120)
ml4s.set_css_style('./include/bootstrap.css')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

z = np.linspace(-10,10,100)

def ELU(z):
    if z <= 0: 
        return (2.71828**z -1)
    else:
        return z
ELU = np.vectorize(ELU)
    
plt.plot(z,ELU(z))
plt.xlabel('z')
plt.ylabel('$a(z)$')


# In[10]:


get_ipython().system('git status')


# In[11]:


get_ipython().system('git add ..')


# In[12]:


get_ipython().system('git status')


# In[14]:


get_ipython().system('git commit -m "Exercise 1"')


# In[15]:



ls -al ~/.ssh


# In[ ]:




