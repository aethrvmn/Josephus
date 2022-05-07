import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# This is the case for k = 2
n, k = np.arange(1, 900), 2

def bruteforce(n, k):
    vector = np.arange(1, n+1)
    while len(vector[vector!=0])>1:
        for i in np.arange(len(vector)-1):
            if vector[i] != 0:
                for j in np.arange(1,n):
                    if vector[i+j] != 0:
                        vector[i+j] = 0
                        break
        if vector[-1] != 0:
            vector[0] = 0
        vector = vector[vector>0]
    return int(vector[0])

def recursion(n, k):

    if n == 1:
        return 1
    else:
        return (recursion(n - 1, k)+k-1)%n + 1

def coprimes(n):
    prepow = np.power(2, np.floor(np.log(n)/np.log(2)))
    nextpow = np.power(2, np.ceil(np.log(n)/np.log(2)))

    num_people = np.array([])
    survivors = np.array([])

    for i in np.arange(prepow, nextpow):
        num_people = np.append(num_people, i)
        v= 2*i-1
        survivors = np.append(survivors, v)

    return survivors[num_people==n]

def gensol(n):
    i = 0
    while n >= 2**i:
        i=i+1
    val = n-2**(i-1)
    return (2*val)+1

def trick(n):
    win = list(str(bin(n)[2:]))
    win = np.append(win, win[0])
    win = np.delete(win, 0)
    win = int(''.join(win), 2)
    return win

brutecomp = np.array([])
recomp = np.array([])
cocomp = np.array([])
gencomp = np.array([])
tricomp = np.array([])
sol = np.array([])

start = time.time()
for i in tqdm(n):
    bruteforce(int(i), k)
    brutecomp = np.append(brutecomp, time.time()-start)

for i in n:
    sol = np.append(sol, gensol(int(i)))
solution = pd.DataFrame({'Solution' :sol}, index = n)
solution.to_csv('solutions.csv')

start = time.time()
for i in tqdm(n):
    recursion(int(i), k)
    recomp = np.append(recomp, time.time()-start)

start = time.time()
for i in tqdm(n):
    coprimes(int(i))
    cocomp = np.append(cocomp, time.time()-start)

start = time.time()
for i in tqdm(n):
    gensol(int(i))
    gencomp = np.append(gencomp, time.time()-start)

start = time.time()
for i in tqdm(n):
    trick(int(i))
    tricomp = np.append(tricomp, time.time()-start)

df = pd.DataFrame({
                    'BruteForce': brutecomp,
                    'CoprimeDecomposition': cocomp,
                    'GeneralSolution': gencomp,
                    'BinaryTrick': tricomp
                    })

df.to_csv('times_w_r.csv')

plt.title('Time Comparison Between Algorithms')
plt.plot(n, brutecomp)
plt.plot(n, recomp)
plt.plot(n, cocomp)
plt.plot(n, gencomp)
plt.plot(n, tricomp)

# plt.yscale('log') # This changes the graph to a log-graph on the y-axis

plt.xlabel('Number $(n)$ of steps')
plt.ylabel('Time')
plt.legend(['Brute Force', 'Recursion', 'Coprime Decomposition', 'General Solution', 'Binary Trick', '$x^{2}$', '$x^{3}$'])

plt.savefig('time_log.png')
plt.show()
