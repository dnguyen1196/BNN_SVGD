import matplotlib.pyplot as plt
import re
import numpy as np


text = """[0.04798984] $ 
[-0.19371149] $ 
[1.8766255] $ 
[0.03175441] $ 
[4.256425] $ 
[-0.2668256] $ 
[-0.08647793] $ 
[-0.26767507] $ 
[2.4665055] $ 
[0.09256556] $ 
[-0.05258361] $ 
[-0.30291903] $ 
[-0.24182485] $ 
[-0.07166407] $ 
[-1.8424444] $ 
[-4.6733413] $ 
[0.40577948] $ 
[0.04084044] $ 
[-0.0115935] $ 
[-2.626696] $ 
"""

dim = 1

pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

data_weight = []
data_bias = []

for line in text.split("\n"):
    s = line.split("$")
    if len(s) < 2:
        break
    weight = s[0]
    bias = s[1]

    result_weight = re.findall(r"[-+]?\d*\.\d+|\d+", weight)
    result_bias   = re.findall(r"[-+]?\d*\.\d+|\d+", bias)

    if result_weight:
        coords = [float(s) for s in result_weight]
        biases = [float(s) for s in result_bias]
        data_weight.append(coords)
        data_bias.append(biases)

data_weight = np.asarray(data_weight)
data_bias = np.asarray(data_bias)

# print(data)
plt.plot(data_weight, np.zeros_like(data_weight), 'x')
plt.show()

plt.plot(data_bias, np.zeros_like(data_bias), 'x')
plt.show()