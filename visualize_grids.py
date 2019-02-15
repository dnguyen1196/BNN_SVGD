import matplotlib.pyplot as plt
import re
import matplotlib.image as mpimg
import numpy as np


ll_sigmas = [0.01, 0.1, 1, 2]
p_sigmas = [0.01, 0.1, 1, 2]
rbf_sigmas = [0.01, 0.1, 1, 2]

folder = "experiments/synthetic/SVGD/"


# fig, axes = plt.subplots(nrows=8, ncols=8)

fig = plt.figure()
fig.set_size_inches(40, 40)

i = 0
for l in ll_sigmas:
    for p in p_sigmas:
        for r in rbf_sigmas:
            a = fig.add_subplot(8, 8, i+1)

            title = "ll_sigma={},prior_sigma={},rbf_sigma={}.png".format(l, p, r)
            img = mpimg.imread(folder + title)

            imgplot = plt.imshow(img)
            a.set_title("ll={},p={},rbf={}.png".format(l, p, r))

            i+= 1

plt.tight_layout()
# plt.show()
plt.savefig("SVGD_64grids.png")