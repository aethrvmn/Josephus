import numpy as np
import time
import pandas as pd
from tqdm import tqdm

n = np.arange(1, 5000)

manual = pd.read_csv('solutions.csv', index_col = 0)

run_time = np.array([])
start = time.time()
for i in tqdm(n):
    manual.Solution[i]
    run_time = np.append(run_time, time.time()-start)

comp = pd.read_csv('complexity.csv', index_col = 0)
complexity = pd.DataFrame({'ReadingFromList': run_time}, index = n)
result = comp.join(complexity)
result.to_csv('complexity.csv')
