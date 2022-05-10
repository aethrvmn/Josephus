import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# We will study the case for k=2
n, k = np.arange(1, 5000), 2

def bruteforce(x, y):
    vector = np.arange(1, x+1)
    while len(vector[vector!=0])>1:
        for i in np.arange(len(vector)-1):
            if vector[i] != 0:
                for j in np.arange(y-1, len(vector), y):
                    if vector[j] != 0:
                        vector[j] = 0
                        break
        if vector[-1] != 0:
            vector[0] = 0
        vector = vector[vector>0]
    return int(vector[0])

run_time = np.array([])
start = time.time()

for i in tqdm(n):
    bruteforce(int(i), k)
    run_time = np.append(run_time, time.time()-start)

complexity = pd.DataFrame({'BruteForce' : run_time}, index = n)
complexity.to_csv('5k_complexity.csv', sep = ',')
