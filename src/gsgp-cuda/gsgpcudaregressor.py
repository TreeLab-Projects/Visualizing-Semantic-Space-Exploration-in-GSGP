import gs
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
import random
import numpy as np
import statistics as stats

# Load data of problem 
path =  "/media/turing/Respaldo/graficas/train_10107_30.txt"
df = pd.read_csv(path, header=None, sep='\s+')
    
# Define row and columns of dataset
nrow = len(df.index)
nvar = df.shape[1]
#Separate data of target
X = df.iloc[0:nrow, 0:nvar-1]
#load colum of target
y = df.iloc[:nrow, nvar-1]
est = gs.GSGPCudaRegressor(
        g=200,
        pop_size=64,
        max_len=1024,
        func_ratio=0.5,
        variable_ratio=0.5,
        max_rand_constant=10,
        sigmoid = 1,
        error_function=0,
        oms=0,
        normalize=0,
        do_min_max=0,
        protected_division=1,
        visualization=0

    )

        
    # Split data 
n = random.randint(0,9000)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.70,test_size=0.30,random_state=n)
    
est.train_and_evaluate_model(X_train, y_train, X_test, y_test)
