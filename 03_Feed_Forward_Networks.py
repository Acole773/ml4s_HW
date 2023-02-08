#!/usr/bin/env python
# coding: utf-8

# # Physics 494/594
# ## Building a Feed Forward Neural Network
# 

# In[4]:


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


# ## Last Time
# 
# ### [Notebook Link: 02_NN_Structure_Feed_Forward.ipynb](./02_NN_Structure_Feed_Forward.ipynb)
# 
# - Built our first neural network
# - randomly assigned weights and biases
# - performed activiations one layer at a time
# 
# ## Today
# 
# - Write code to propagate activations through layers
# - Manually 'train' to discern features

# ### Recall our 3x3 picture
# 
# I've defined a function `print_rectangle(...)` that will allows for code resuse.  This is a great programming practice!

# In[5]:


L = 3
N0 = L*L
x = [0,0,0,1,1,0,1,1,0]

def print_rectangle(x):
    print(''.join([ci if (i+1)%L else ci+'\n' for i,ci in 
                 enumerate([' ▉ ' if cx else ' ░ ' for i,cx in enumerate(x)])]))
print_rectangle(x)


# In[13]:


def print_rectangle_1(x):
    L = int(np.sqrt(len(x)))
    print(*[''.join(i) for i in np.array([' ▉ ',' ░ '])[x].reshape(L,L)],sep='\n')


# In[6]:


g = np.array([[1,1,1],[0,0,0],[2,2,2]])
np.sum(g,axis=1)


# In[7]:


print_rectangle(x)


# In[8]:


ml4.draw_network([9,2,1])


# 
# ## Feed Forward
# 
# Previously we manually propagated activations through a deep neural network one layer at a time. 
# 
# Recall, that for a single layer: 
# \begin{align}
# a_j^\ell &= \sigma\left(\boldsymbol{z}^{\ell}\right) \\
# &= \sigma \left(\sum_k w_{jk}^\ell a_k^{\ell-1} + b_j^\ell \right) \\
#  &= \sigma\left(\boldsymbol{\mathsf{w}}^\ell \cdot \boldsymbol{a}^{\ell-1} + \boldsymbol{b}^\ell\right)
# \end{align}
# 
# Given the values in the input layer $\boldsymbol{x} \equiv \boldsymbol{a}^0$, and all weights and biases, we want to compute $\boldsymbol{z}^{\ell}$, apply the activations sequentially to each layer, and return the output of the entire network.

# In[17]:


def feed_forward(a0,w,b):
    ''' Compute the output of a deep neural network given the input (a0) 
        and the weights (w) and biaes (b).
    '''
    a = a0
    num_layers = len(b)
    
    # feed input layer forward
    for ℓ in range(num_layers):
        z = w[ℓ] @ a + b[ℓ] #reuse z for every step
        a = 1.0/(1.0 + np.exp(-z))
    return a


# Next, we will randomly set all the weights and biases for the 1 hidden and 1 output layer of our network.  We used a hidden-layer with only 2 neurons, feel free to change this when  you are working on your notebook.

# In[39]:


N = [9,2,1]
w,b = [],[]

# append to the weights and biases list.  Make sure you get the dimensions correct!
for ℓ in range(1,len(N)): #simple for loop
    w.append(np.random.uniform(low=-10,high=10,size=(N[ℓ],N[ℓ-1])))
    b.append(np.random.uniform(low=-1,high=1, size=N[ℓ]))


# In[28]:


print(w)
len(w)


# In[29]:



w[0].shape


# Let's compute (and output) the activation of the output layer. 
# 
# We can keep randomly generating new weights and biases (by executing the code above) until we find a set that is close to 1 (which we want for our rectangle)

# In[40]:


feed_forward(x,w,b)


# ### Visualize the Final Network:

# In[41]:


ml4s.draw_network(N, weights=w, biases=b, node_labels=[])


# <div class="span alert alert-success">
# <h4> Excercises </h4>
# <ol>
#     <li>Find the output from the neural network for the following inputs 
#         <p>
#             <code>x = [1,1,1,0,0,0,0,0,0]</code> <br />
#             <code>x = [1,0,0,0,1,0,0,0,1]</code> <br />
#             <code>x = [0,0,0,0,0,0,0,0,1]</code> <br />
#         </p>
#        You can use the <code>print_rectangle(x)</code> function to visualize.
#     </li>
#     <li> Modify your <code>feed_forward</code> function to use a ReLU instead of a sigmoid.  Are there any changes?
#     </li>
# </ol>
# </div>

# In[ ]:


#Excercises Part 1: 


# In[43]:


L = 3
N0 = L*L
x = [1,1,1,0,0,0,0,0,0] # diffine new x array

def print_rectangle(x):
    print(''.join([ci if (i+1)%L else ci+'\n' for i,ci in 
                 enumerate([' ▉ ' if cx else ' ░ ' for i,cx in enumerate(x)])]))
print_rectangle(x) #and visualize, To me this looks like a rectangle.


# In[44]:


feed_forward(x,w,b) # and yes my weights and biases agree that it is rectangular. 


# In[45]:


ml4s.draw_network(N, weights=w, biases=b, node_labels=[]) 


# In[46]:


L = 3
N0 = L*L
x = [1,0,0,0,1,0,0,0,1] # diffine new x array

def print_rectangle(x):
    print(''.join([ci if (i+1)%L else ci+'\n' for i,ci in 
                 enumerate([' ▉ ' if cx else ' ░ ' for i,cx in enumerate(x)])]))
print_rectangle(x) #and visualize, To me this does not look like a rectangle


# In[47]:


feed_forward(x,w,b) # and yes my weights and biases agree that it is not rectangular.


# In[49]:


L = 3
N0 = L*L
x = [0,0,0,0,0,0,0,0,1] # diffine new x array

def print_rectangle(x):
    print(''.join([ci if (i+1)%L else ci+'\n' for i,ci in 
                 enumerate([' ▉ ' if cx else ' ░ ' for i,cx in enumerate(x)])]))
print_rectangle(x) #and visualize, To me this is a square, which I think is a rectangle


# In[50]:


feed_forward(x,w,b) #so my weights and biases are not good at identifying squares


# In[52]:


#Excercises Part 2: 


# In[62]:


# I go back to original x 
L = 3
N0 = L*L
x = [0,0,0,1,1,0,1,1,0]

def print_rectangle(x):
    print(''.join([ci if (i+1)%L else ci+'\n' for i,ci in 
                 enumerate([' ▉ ' if cx else ' ░ ' for i,cx in enumerate(x)])]))
print_rectangle(x)


# In[9]:


def feed_forward2(a0,w,b):
    ''' Compute the output of a deep neural network given the input (a0) 
        and the weights (w) and biaes (b).
    '''
    a = a0
    num_layers = len(b)
    
    # feed input layer forward
    for ℓ in range(num_layers):
        z = w[ℓ] @ a + b[ℓ] #reuse z for every step
        α = 1.0/10
        a = np.maximum(α*z, z)
    return a


# In[64]:


feed_forward2(x,w,b)


# In[1]:


# I note with the same values of weights and biaes I get a large number, 
# this is confusing because I expect result to be between 0 and 1.
# let me try with other x


# In[2]:


L = 3
N0 = L*L
x = [1,0,0,0,1,0,0,0,1] # diffine new x array

def print_rectangle(x):
    print(''.join([ci if (i+1)%L else ci+'\n' for i,ci in 
                 enumerate([' ▉ ' if cx else ' ░ ' for i,cx in enumerate(x)])]))
print_rectangle(x) #and visualize, To me this does not look like a rectangle


# In[10]:


feed_forward2(x,w,b)


# In[ ]:


# I had an auto save happen when I was working here and I lost my values of weights and biaes
# so I will stop here because I don't want toruin my previous work.

