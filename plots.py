import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('complexity.csv', index_col = 0)

df.plot(
        kind = 'line',
        title = 'Time Comparison Between Algorithms',
        xlabel = 'Number $n$ of Iterations',
        ylabel = 'Time',
        # logy = True
        )

plt.savefig('time.png')

df[['CoprimeDecomposition', 'ReadingFromList', 'GeneralSolution', 'BinaryTrick']].plot(
        kind = 'line',
        title = 'Time Comparison Between General Solution and the Binary Trick',
        xlabel = 'Number $n$ of Iterations',
        ylabel = 'Time',
        # logy = True
        )

plt.savefig('time2.png')
