import numpy as np
import time
import pandas as pd
from tqdm import tqdm

n, k = np.arange(1, 5000), 2

def coprimes(n):
    prepow = np.power(2, np.floor(np.log(n)/np.log(2)))
    nextpow = np.power(2, np.ceil(np.log(n)/np.log(2)))
    if prepow == nextpow:
        return 1
    else:
        num_people = np.array([])
        survivors = np.array([])

        for i in np.arange(prepow, nextpow):
            num_people = np.append(num_people, i)
        for j in np.arange(1, nextpow-prepow+1):
            v= 2*j-1
            survivors = np.append(survivors, v)

        return survivors[num_people==n]

run_time = np.array([])
start = time.time()

for i in tqdm(n):
    coprimes(int(i))
    run_time = np.append(run_time, time.time()-start)

comp = pd.read_csv('complexity.csv', index_col = 0)
complexity = pd.DataFrame({'CoprimeDecomposition': run_time}, index = n)
result = comp.join(complexity)
result.to_csv('complexity.csv')
