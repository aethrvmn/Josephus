import pandas as pd
import matplotlib.pyplot as plt

n = 2000

complexity = pd.read_csv(f"csvs/{n}_complexity.csv", header = 0)

complexity.plot(
        kind = 'line',
        title = 'Time Comparison Between Algorithms',
        xlabel = 'Number $n$ of Iterations',
        ylabel = 'Time',
        figsize = (12,7),
        logy = True,
        )

plt.savefig(f"plots/{n}_logplot.png")

plt.show()
