import numpy as np
import time
import pandas as pd
from tqdm import tqdm

n, k = np.arange(1, 9000), 2

manual = pd.read_csv('solutions.csv', index_col = 0)

run_time = np.array([])
start = time.time()
for i in tqdm(n):

    run_time = np.append(run_time, time.time()-start)

comp = pd.read_csv('complexity.csv', index_col = 0)
complexity = pd.DataFrame({'Stack': run_time}, index = n)
result = comp.join(complexity)
result.to_csv('complexity.csv')
