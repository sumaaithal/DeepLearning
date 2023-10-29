import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importing the dataset

movies = pd.read_csv("C:\\Users\\SKUP\\OneDrive - Capco\\Desktop\\AI-ML\\data_AI_ML\\deep learning\\P16-Boltzmann-Machines\\Boltzmann_Machines\\ml-1m\\movies.dat",
                     sep='::', header = None, engine='python', encoding='latin-1')

users = pd.read_csv("C:\\Users\\SKUP\\OneDrive - Capco\\Desktop\\AI-ML\\data_AI_ML\\deep learning\\P16-Boltzmann-Machines\\Boltzmann_Machines\\ml-1m\\users.dat",
                     sep='::', header = None, engine='python', encoding='latin-1')

ratings = movies = pd.read_csv("C:\\Users\\SKUP\\OneDrive - Capco\\Desktop\\AI-ML\\data_AI_ML\\deep learning\\P16-Boltzmann-Machines\\Boltzmann_Machines\\ml-1m\\ratings.dat",
                     sep='::', header = None, engine='python', encoding='latin-1')

#preparing the training & test set
training_set = pd.read_csv("C:\\Users\\SKUP\\OneDrive - Capco\\Desktop\\AI-ML\\data_AI_ML\\deep learning\\P16-Boltzmann-Machines\\Boltzmann_Machines\\ml-100k\\u1.base",
                           delimiter='\t',  )
training_set = np.array(training_set, dtype='int' )

test_set = pd.read_csv("C:\\Users\\SKUP\\OneDrive - Capco\\Desktop\\AI-ML\\data_AI_ML\\deep learning\\P16-Boltzmann-Machines\\Boltzmann_Machines\\ml-100k\\u1.test",
                           delimiter='\t',  )
test_set = np.array(test_set, dtype='int' )

## creating number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#convert training and test set into a matrix 
def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:,1][data[:,0]==id_users]
        id_ratings = data[:,2][data[:,0]==id_users]
        ratings = np.zeros(nb_movies)
        
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
        
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)


## convert data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)










