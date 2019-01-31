import matplotlib.pyplot as plt
import re
import numpy as np


text = """[0.7887073] $ 
[2.9651] $ 
[1.3373252] $ 
[1.9351485] $ 
[-3.600162] $ 
[2.1068032] $ 
[1.7039715] $ 
[0.5915569] $ 
[2.811709] $ 
[0.67247176] $ 
[1.889285] $ 
[5.1111817] $ 
[1.5320947] $ 
[1.9065022] $ 
[0.82902914] $ 
[2.8724341] $ 
[1.4443517] $ 
[2.4661376] $ 
[1.6679454] $ 
[2.317671] $ 
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