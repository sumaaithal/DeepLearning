import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import *
from pylab import bone,colorbar,pcolor,plot, show
import tensorflow as tf
tf.__version__

## import dataset
data = pd.read_csv("C:\\Users\\SKUP\\OneDrive - Capco\\Desktop\\AI-ML\\data_AI_ML\\notebooks\\datasets\\Credit_Card_Applications.csv")


X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

## feature scaling
sc = MinMaxScaler(feature_range=(0,1))
X_scaled = sc.fit_transform(X)

## training som
som = MiniSom(x=10, y=10, input_len=15,random_seed=42)
som.random_weights_init(X)
som.train_random(X, 100)

#visuals
bone()
pcolor(som.distance_map().T) #returns mean interneuron distances
colorbar()
markers = ['o','s']
colors = ['r','g']

for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5, 
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth=2)

show()

#finding frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,3)],mappings[(8,3)]), axis=0 )

frauds = sc.inverse_transform(frauds)
print(frauds)

## part 2 going from unsupervised to supervised

##creating matrix of features
customers = data.iloc[:,1:].values

#create a dependent variable - whether fraud or not
#initialize vector

is_fraud = np.zeros(len(data))
for i in range(len(data)):
    if data.iloc[i,0] in frauds:
        is_fraud[i] = 1
        
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=2, activation='relu', input_dim=15, kernel_initializer='uniform' ))


# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid',kernel_initializer='uniform'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

# Predicting the probabilities of fraud
y_pred = ann.predict(customers)   
        
y_pred = np.concatenate( (data.iloc[:,0:1].values, y_pred) , axis=1 )

y_pred = y_pred[y_pred[:,1].argsort()]
        
        
    
    
    