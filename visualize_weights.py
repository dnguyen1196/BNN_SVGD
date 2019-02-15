import matplotlib.pyplot as plt
import re
import numpy as np


text = """[-0.1654381] [-6.975857]
[-0.8497792] [-1.1237332]
[-6.4340744] [-0.15945515]
[-0.1256136] [-9.3685665]
[8.137452] [0.10550562]
[-1.8540217] [-0.53344655]
[3.6573265] [0.27838865]
[-0.43437067] [-2.309113]
[-8.833587] [-0.11749676]
[0.32436582] [3.1324208]
[0.1928124] [4.9977007]
[1.9321207] [0.5135509]
[0.6610596] [1.4770778]
[0.09765913] [9.808946]
[-4.5688252] [-0.22357889]
[-3.122946] [-0.3253422]
[5.6536713] [0.1617065]
[-0.20127551] [-5.079725]
[0.13150403] [7.1642156]
[-0.28493074] [-3.5762618]
"""

dim = 1

pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

data_weight = []
data_bias = []

for line in text.split("\n"):
    result_weight = re.findall(r"[-+]?\d*\.\d+|\d+", line)

    if result_weight:
        coords = [float(s) for s in result_weight]
        data_weight.append(coords)


data_weight = np.asarray(data_weight)
print(data_weight.shape)

# print(data)
plt.scatter(data_weight[:,0], data_weight[:,1])
plt.show()
