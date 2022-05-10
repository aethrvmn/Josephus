import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brute, coprimes, stack, general, binary, read

df = pd.read_csv('5k_complexity.csv', index_col = 0)

df.plot(
        kind = 'line',
        title = 'Time Comparison Between Algorithms',
        xlabel = 'Number $n$ of Iterations',
        ylabel = 'Time',
        logy = True
        )

plt.savefig('5k_time_log.png')

df.plot(
        kind = 'line',
        title = 'Time Comparison Between General Solution and the Binary Trick',
        xlabel = 'Number $n$ of Iterations',
        ylabel = 'Time',
        # logy = True
        )

plt.savefig('5k_time.png')
plt.show()
