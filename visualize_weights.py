import matplotlib.pyplot as plt
import re
import numpy as np


text = """[-1.4301972] [-1.1117795]
[-1.6243254] [-0.7688905]
[-1.4003739] [-1.2917042]
[-1.7050871] [-0.9684799]
[1.0005841] [-0.99352276]
[1.526458] [1.4883348]
[0.890705] [-0.98977816]
[-1.9910933] [-0.74045074]
[0.19013226] [-0.45090926]
[0.9518597] [1.3194367]
[0.26199567] [-0.29698992]
[-1.0582031] [-1.8261899]
[1.4022833] [1.3904434]
[-0.7790705] [-1.5470351]
[-0.6075901] [-1.839501]
[-1.7850478] [-0.54680645]
[0.8388847] [1.8881936]
[1.2668731] [1.5318996]
[-1.9127779] [-1.4809644]
[-1.8298194] [-1.5875113]
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


refx_neg = np.linspace(-10, -0.1, 19)
refx_pos = np.linspace(0.1, 10, 19)
refy_neg = 1. / refx_neg
refy_pos = 1. / refx_pos

plt.plot(refx_neg, refy_neg, "r")
plt.plot(refx_pos, refy_pos, "r")


plt.show()
