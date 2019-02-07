import matplotlib.pyplot as plt
import re
import numpy as np


text = """-12.332584 -0.095041424
-0.083616145 -6.5055194
11.510701 0.08563855
-0.18394023 -5.375379
-2.2417085 -0.641133
-0.6946191 -1.5224769
0.29379037 4.955253
1.372003 0.5627461
-0.1259807 -10.203814
-0.32535356 -2.7515426
-0.28805175 -3.918028
-7.4598656 -0.11256498
-4.0735197 -0.22213012
0.5975713 0.664911
0.3261268 2.4364712
8.2039385 0.13236411
-1.1201141 -0.74418235
4.7340655 0.25268292
5.8182116 0.19871303
-4.8712835 -0.11543308
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
