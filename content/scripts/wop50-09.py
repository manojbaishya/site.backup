#! /usr/bin/env python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

wop = pd.read_csv("../data/wop50-09.csv")

ax = sns.regplot(
    data=wop,
    x="year",
    y="prod-in-mil-barrels-per-day",
    marker="o",
    scatter_kws={"facecolors": "none", "edgecolors": "green"},
)
ax.set(xlabel="Year", ylabel="Production in Million Barrels Per Day", title="World Oil Production")
plt.savefig("../data/wop50-09.svg", format="svg", dpi=96)
