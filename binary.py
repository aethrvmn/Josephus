import numpy as np
import time
import pandas as pd
from tqdm import tqdm

n, k = np.arange(1, 5000), 2

def trick(n):
    win = list(str(bin(n)[2:]))
    win = np.append(win, win[0])
    win = np.delete(win, 0)
    win = int(''.join(win), 2)
    return win

run_time = np.array([])
start = time.time()

for i in tqdm(n):
    trick(int(i))
    run_time = np.append(run_time, time.time()-start)

comp = pd.read_csv('5k_complexity.csv', index_col = 0)
complexity = pd.DataFrame({'BinaryTrick': run_time}, index = n)
result = comp.join(complexity)
result.to_csv('5k_complexity.csv')
