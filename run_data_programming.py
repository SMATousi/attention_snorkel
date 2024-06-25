#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensordict.tensordict import TensorDict
import sys
import wandb


# # Creating Data

# In[2]:


np.random.seed(42)


X, y = make_classification(n_samples=200000, n_features=3, n_informative=3, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.5, 0.5], flip_y=0.05, class_sep=1.5)
y = 2*y - 1

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')


# ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], X[y == -1][:, 2], c='b', marker='o', label='Class -1')
# ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], c='r', marker='^', label='Class 1')

# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Feature 3')
# ax.set_title('3D Scatter Plot of Synthetic Data')
# ax.legend()

# plt.show()


# # LFs generator

# In[3]:


def random_label_flip_and_zero(arr, m, n_list, zero_n_list):

    if len(n_list) != m or len(zero_n_list) != m:
        raise ValueError("The length of n_list and zero_n_list must be equal to m.")
    
    length = len(arr)
    flipped_arrays = []

    for i in range(m):
        n = n_list[i]
        zero_n = zero_n_list[i]

        # Randomly select indices to flip
        indices_to_zero = np.random.choice(length, zero_n, replace=False)

        # Create a copy of the array to flip the labels
        modified_arr = arr.copy()
        modified_arr[indices_to_zero] = 0

        # Identify the untouched indices
        untouched_indices = np.setdiff1d(np.arange(length), indices_to_zero)

        # Randomly select indices from the untouched indices to set to 0
        indices_to_flip = np.random.choice(untouched_indices, n, replace=False)

        # Set the chosen indices to 0
        modified_arr[indices_to_flip] = -modified_arr[indices_to_flip]

        flipped_arrays.append(modified_arr)

    return flipped_arrays


# In[4]:


arr = y

m = 5  


beta_list = [0.4 for i in range(m)]
zero_n_list = [int((1 - beta)* y.shape[0]) for beta in beta_list]  

alpha_list = [0.89 for i in range(m)]
n_list = [int((1-alpha)*(y.shape[0] - zero_n_list[i])) for i, alpha in enumerate(alpha_list)] 

# m = 5
# beta_list = [0.35, 0.39, 0.42, 0.46, 0.5]
# zero_n_list = [int((1 - beta)* y.shape[0]) for beta in beta_list]  

# alpha_list = [0.81, 0.82, 0.84, 0.86, 0.90]
# n_list = [int((1-alpha)*(y.shape[0] - zero_n_list[i])) for i, alpha in enumerate(alpha_list)] 

# print(n_list)

flipped_arrays = random_label_flip_and_zero(arr, m, n_list, zero_n_list)


# print("Original = ", arr)
ALL_LFs = {}

for i, modified_arr in enumerate(flipped_arrays):
#     print(f"Array {i+1}:")
#     print(modified_arr-arr)
    lf_dict = {}
    
    lf_dict['alpha'] = 1 - (n_list[i]/(len(y) - zero_n_list[i]))
    lf_dict['beta'] = 1 - (zero_n_list[i]/len(y))
    
    lf_dict['outputs'] = modified_arr
    
    ALL_LFs[i] = lf_dict


# In[5]:


ALL_LFs


# # Expected Value for alpha and beta

# In[6]:


m = 5
epsilon = 0.1
s_cardinality = len(y)

minimum_cardinality = (356/(epsilon)**2) * np.log(m/(3*epsilon))

print("minimum cardinality = ", minimum_cardinality)
print("current cardinality = ", s_cardinality)
if s_cardinality > minimum_cardinality:
    print("Check!")
else:
    print("More data needed ...")


# # Label Model

# In[7]:


# initializing

Alpha_Beta_numpy = np.random.rand(m,2)
Alpha_Beta = torch.tensor(Alpha_Beta_numpy, requires_grad=True)

class LabelModel(nn.Module):
    def __init__(self):
        super(LabelModel, self).__init__()
#         self.sigmoid = torch.sigmoid()
        Alpha_Beta_numpy = np.random.rand(m,2)
        self.alpha_beta_array = nn.Parameter(torch.tensor(Alpha_Beta_numpy, requires_grad=True))
        
    def forward(self, lf_label, true_label):
        
        all_lf_probls = 1
        
        for lf_index in range(self.alpha_beta_array.shape[0]):
            
            lf_alpha = torch.sigmoid(self.alpha_beta_array[lf_index,0])
            lf_beta = torch.sigmoid(self.alpha_beta_array[lf_index,1])
            
            if lf_label[lf_index] == true_label:
                
                lf_prob = lf_alpha * lf_beta
            
            if lf_label[lf_index] == -true_label:
                
                lf_prob = (1 - lf_alpha) * lf_beta
            
            if lf_label[lf_index] == 0:
                
                lf_prob = 1 - lf_beta
        
            all_lf_probls = all_lf_probls * lf_prob
        
        
        return 0.5 * all_lf_probls


# In[ ]:


model = LabelModel()


# In[ ]:


model(alpha_beta_array=test_lf,
      lf_label=test_lf_label,
      true_label=test_label)


# # Data Loader

# In[8]:


class LF_Output_Dataset(Dataset):
    def __init__(self, ALL_LFs, X):
        
        self.ALL_LFs = ALL_LFs
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        data_sample = self.X[idx]
        lf_outputs = []
        
        for key in self.ALL_LFs.keys():
            
            lf_outputs.append(self.ALL_LFs[key]['outputs'][idx])
        
        return data_sample, lf_outputs


# In[9]:


dataset = LF_Output_Dataset(ALL_LFs, X)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


# In[ ]:





# # Training Loop

# In[ ]:





# In[ ]:


wandb.init(
        # set the wandb project where this run will be logged
    project='Snorkel-Repro', name='Data-200k-epochs-1000-m-5-alpha89-beta40-lre-5'

        # track hyperparameters and run metadata
        # config={
        # "learning_rate": 0.02,
        # "architecture": "CNN",
        # "dataset": "CIFAR-100",
        # "epochs": 20,
        # }
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LabelModel()

    
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

num_epochs = 100

for epoch in range(num_epochs):
    
    initial_loss = 0
    optimizer.zero_grad()
    
    for data_sample, lf_outputs in tqdm(data_loader):
        
#         lf_outputs = [par.to(device) for par in lf_outputs]
        
        marginal_prob = model(lf_outputs, true_label=1) + model(lf_outputs, true_label=-1)
        log_marginal_prob = torch.log(marginal_prob)
        initial_loss = initial_loss + log_marginal_prob
        
        

        
#         break
    
    initial_loss = -initial_loss
    initial_loss.backward()
    optimizer.step()
    
    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Loss = {-initial_loss.item()}")
        
        for param in model.parameters():
            tensor_dict = {f'Params/tensor_{i}_{j}': torch.sigmoid(param[i, j]).item() for i in range(param.size(0)) for j in range(param.size(1))}
            wandb.log({"Prob/Prob":-initial_loss.item()})
            wandb.log(tensor_dict)
#             print(torch.sigmoid(param),flush=True)

        


# In[ ]:


param[0,1]


# In[ ]:




