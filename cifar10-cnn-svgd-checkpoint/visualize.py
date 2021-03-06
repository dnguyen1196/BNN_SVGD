import re
import matplotlib.pyplot as plt
import numpy as np

input = """Epoch 0, lr = 1.0000
epoch 0, train_model-accuracy = 0.19366, evaluate_test_set-accuracy = 0.2234
Total time since training starts:  699.0787
Epoch 1, lr = 0.9000
epoch 1, train_model-accuracy = 0.28406, evaluate_test_set-accuracy = 0.3163
Total time since training starts:  1389.0819
Epoch 2, lr = 0.8100
epoch 2, train_model-accuracy = 0.34734, evaluate_test_set-accuracy = 0.3639
Total time since training starts:  2076.4731
Epoch 3, lr = 0.7290
epoch 3, train_model-accuracy = 0.42338, evaluate_test_set-accuracy = 0.4113
Total time since training starts:  2768.372
Epoch 4, lr = 0.6561
epoch 4, train_model-accuracy = 0.47294, evaluate_test_set-accuracy = 0.3906
Total time since training starts:  3466.1299
Epoch 5, lr = 0.5905
epoch 5, train_model-accuracy = 0.51114, evaluate_test_set-accuracy = 0.4498
Total time since training starts:  4162.6169
Epoch 6, lr = 0.5314
epoch 6, train_model-accuracy = 0.5485, evaluate_test_set-accuracy = 0.4941
Total time since training starts:  4851.8159
Epoch 7, lr = 0.4783
epoch 7, train_model-accuracy = 0.58138, evaluate_test_set-accuracy = 0.5133
Total time since training starts:  5544.1093
Epoch 8, lr = 0.4305
epoch 8, train_model-accuracy = 0.6085, evaluate_test_set-accuracy = 0.5059
Total time since training starts:  6236.8206
Epoch 9, lr = 0.3874
epoch 9, train_model-accuracy = 0.63238, evaluate_test_set-accuracy = 0.5159
Total time since training starts:  6925.8561
Epoch 10, lr = 0.3487
epoch 10, train_model-accuracy = 0.65682, evaluate_test_set-accuracy = 0.5076
Total time since training starts:  7618.7609
Epoch 11, lr = 0.3138
epoch 11, train_model-accuracy = 0.68264, evaluate_test_set-accuracy = 0.573
Total time since training starts:  8309.9874
Epoch 12, lr = 0.2824
epoch 12, train_model-accuracy = 0.70312, evaluate_test_set-accuracy = 0.5634
Total time since training starts:  9003.1032
Epoch 13, lr = 0.2542
epoch 13, train_model-accuracy = 0.72116, evaluate_test_set-accuracy = 0.5846
Total time since training starts:  9690.8439
Epoch 14, lr = 0.2288
epoch 14, train_model-accuracy = 0.7397, evaluate_test_set-accuracy = 0.6066
Total time since training starts:  10380.4707
Epoch 15, lr = 0.2059
epoch 15, train_model-accuracy = 0.75578, evaluate_test_set-accuracy = 0.6077
Total time since training starts:  11073.0971
Epoch 16, lr = 0.1853
epoch 16, train_model-accuracy = 0.76932, evaluate_test_set-accuracy = 0.6027
Total time since training starts:  11762.1644
Epoch 17, lr = 0.1668
epoch 17, train_model-accuracy = 0.78342, evaluate_test_set-accuracy = 0.6155
Total time since training starts:  12455.3093
Epoch 18, lr = 0.1501
epoch 18, train_model-accuracy = 0.79334, evaluate_test_set-accuracy = 0.597
Total time since training starts:  13152.1299
Epoch 19, lr = 0.1351
epoch 19, train_model-accuracy = 0.80496, evaluate_test_set-accuracy = 0.6216
Total time since training starts:  13848.4325
Epoch 20, lr = 0.1216
epoch 20, train_model-accuracy = 0.81474, evaluate_test_set-accuracy = 0.6155
Total time since training starts:  14539.3121
Epoch 21, lr = 0.1094
epoch 21, train_model-accuracy = 0.82252, evaluate_test_set-accuracy = 0.6228
Total time since training starts:  15232.2988
Epoch 22, lr = 0.0985
epoch 22, train_model-accuracy = 0.8314, evaluate_test_set-accuracy = 0.6389
Total time since training starts:  15924.3843
Epoch 23, lr = 0.0886
epoch 23, train_model-accuracy = 0.83758, evaluate_test_set-accuracy = 0.6402
Total time since training starts:  16616.5366
Epoch 24, lr = 0.0798
epoch 24, train_model-accuracy = 0.84308, evaluate_test_set-accuracy = 0.6406
Total time since training starts:  17313.0015
Epoch 25, lr = 0.0718
epoch 25, train_model-accuracy = 0.84774, evaluate_test_set-accuracy = 0.6442
Total time since training starts:  18003.7951
Epoch 26, lr = 0.0646
epoch 26, train_model-accuracy = 0.85258, evaluate_test_set-accuracy = 0.6429
Total time since training starts:  18695.0818
Epoch 27, lr = 0.0581
epoch 27, train_model-accuracy = 0.85664, evaluate_test_set-accuracy = 0.653
Total time since training starts:  19386.0289
Epoch 28, lr = 0.0523
epoch 28, train_model-accuracy = 0.85898, evaluate_test_set-accuracy = 0.653
Total time since training starts:  20080.1807
Epoch 29, lr = 0.0471
epoch 29, train_model-accuracy = 0.86324, evaluate_test_set-accuracy = 0.6479
Total time since training starts:  20774.052
Epoch 30, lr = 0.0424
epoch 30, train_model-accuracy = 0.86552, evaluate_test_set-accuracy = 0.653
Total time since training starts:  21466.0985
Epoch 31, lr = 0.0382
epoch 31, train_model-accuracy = 0.86706, evaluate_test_set-accuracy = 0.6516
Total time since training starts:  22160.7705
Epoch 32, lr = 0.0343
epoch 32, train_model-accuracy = 0.86834, evaluate_test_set-accuracy = 0.6541
Total time since training starts:  22854.1225
Epoch 33, lr = 0.0309
epoch 33, train_model-accuracy = 0.86984, evaluate_test_set-accuracy = 0.6558
Total time since training starts:  23548.3143
Epoch 34, lr = 0.0278
epoch 34, train_model-accuracy = 0.87018, evaluate_test_set-accuracy = 0.6581
Total time since training starts:  24244.6007
Epoch 35, lr = 0.0250
epoch 35, train_model-accuracy = 0.87268, evaluate_test_set-accuracy = 0.6588
Total time since training starts:  24943.4827
Epoch 36, lr = 0.0225
epoch 36, train_model-accuracy = 0.87214, evaluate_test_set-accuracy = 0.6564
Total time since training starts:  25636.7735
Epoch 37, lr = 0.0203
epoch 37, train_model-accuracy = 0.8739, evaluate_test_set-accuracy = 0.6607
Total time since training starts:  26329.709
Epoch 38, lr = 0.0182
epoch 38, train_model-accuracy = 0.87344, evaluate_test_set-accuracy = 0.6589
Total time since training starts:  27024.4828
Epoch 39, lr = 0.0164
epoch 39, train_model-accuracy = 0.87398, evaluate_test_set-accuracy = 0.6586
Total time since training starts:  27714.0686
Epoch 40, lr = 0.0148
epoch 40, train_model-accuracy = 0.8737, evaluate_test_set-accuracy = 0.6602
Total time since training starts:  28416.2551
Epoch 41, lr = 0.0133
epoch 41, train_model-accuracy = 0.87482, evaluate_test_set-accuracy = 0.66
Total time since training starts:  29110.711
Epoch 42, lr = 0.0120
epoch 42, train_model-accuracy = 0.87386, evaluate_test_set-accuracy = 0.6606
Total time since training starts:  29803.4284
Epoch 43, lr = 0.0108
epoch 43, train_model-accuracy = 0.8733, evaluate_test_set-accuracy = 0.6592
Total time since training starts:  30497.4783
Epoch 44, lr = 0.0097
epoch 44, train_model-accuracy = 0.87378, evaluate_test_set-accuracy = 0.6621
Total time since training starts:  31191.5576
Epoch 45, lr = 0.0087
epoch 45, train_model-accuracy = 0.87384, evaluate_test_set-accuracy = 0.6626
Total time since training starts:  31884.1224
Epoch 46, lr = 0.0079
epoch 46, train_model-accuracy = 0.87356, evaluate_test_set-accuracy = 0.661
Total time since training starts:  32579.9278
Epoch 47, lr = 0.0071
epoch 47, train_model-accuracy = 0.87356, evaluate_test_set-accuracy = 0.6623
Total time since training starts:  33278.0141
Epoch 48, lr = 0.0064
epoch 48, train_model-accuracy = 0.87428, evaluate_test_set-accuracy = 0.6626
Total time since training starts:  33970.1996
Epoch 49, lr = 0.0057
epoch 49, train_model-accuracy = 0.87344, evaluate_test_set-accuracy = 0.6614
Total time since training starts:  34664.4183
Epoch 50, lr = 0.0052"""


line_num = 0

train_accuracy = []
test_accuracy  = []
times          = []

for line in input.split("\n"):
    metrics = re.findall(r"[-+]?\d*\.\d+|\d+", line)

    if line_num % 3 == 0:
        lr = metrics[-1]
    elif line_num % 3 == 1:
        train_acc = float(metrics[1]) * 100
        test_acc  = float(metrics[2]) * 100
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
    else:
        time      = int(float(metrics[0]))
        times.append(time)

    line_num += 1

plt.plot(times, train_accuracy)
plt.plot(times, test_accuracy)
plt.legend(["train accuracy", "test accuracy"])
plt.ylim([0, 100])
plt.xlabel("Seconds")

xticks = np.arange(0, 35001, 5000)

plt.xticks(xticks)

plt.show()
