import numpy as np
import time
import pandas as pd
from tqdm import tqdm

n, k = np.arange(1, 5000), 2

def stack(n, k):
    vector=[]
    for i in range(1,n+1):
        vector.append(i)
    if len(vector) == 1:
        return 1
    else:
        i = 0
        while len(vector) > 1:
            i = (i+k-1)%len(vector)
            vector.pop(i)
        return vector[0]

run_time = np.array([])
start = time.time()

for i in tqdm(n):
    stack(int(i), k)
    run_time = np.append(run_time, time.time()-start)

comp = pd.read_csv('5k_complexity.csv', index_col = 0)
complexity = pd.DataFrame({'Stack': run_time}, index = n)
result = comp.join(complexity)
result.to_csv('5k_complexity.csv')
