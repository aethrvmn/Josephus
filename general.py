import numpy as np
import time
import pandas as pd
from tqdm import tqdm

n, k = np.arange(1, 5000), 2

def gensol(n):
    i = 0
    while n >= 2**i:
        i=i+1
    val = n-2**(i-1)
    return (2*val)+1

run_time = np.array([])
start = time.time()

for i in tqdm(n):
    gensol(int(i))
    run_time = np.append(run_time, time.time()-start)

comp = pd.read_csv('complexity.csv', index_col = 0)
complexity = pd.DataFrame({'GeneralSolution': run_time}, index = n)
result = comp.join(complexity)
result.to_csv('complexity.csv')
