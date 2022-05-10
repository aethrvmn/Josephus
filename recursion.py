import numpy as np
import time
import pandas as pd
from tqdm import tqdm

n, k = np.arange(1, 900), 2

def recursion(n, k):
    if n == 1:
        return 1
    else:
        return (recursion(n - 1, k)+k-1)%n + 1

run_time = np.array([])
start = time.time()

for i in tqdm(n):
    recursion(int(i), k)
    run_time = np.append(run_time, time.time()-start)

comp = pd.read_csv('900_complexity.csv', index_col = 0)
complexity = pd.DataFrame({'Recursion': run_time}, index = n)
result = comp.join(complexity)
result.to_csv('900_complexity.csv')
