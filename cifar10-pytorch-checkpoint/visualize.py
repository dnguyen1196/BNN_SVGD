import re
import matplotlib.pyplot as plt
import numpy as np

input = """
Epoch: 0
Epoch 0, train accuracy = 21.5
test accuracy = 27.53
Saving..
Total time since training starts:  32.9929

Epoch: 1
Epoch 1, train accuracy = 33.554
test accuracy = 39.4
Saving..
Total time since training starts:  66.6113

Epoch: 2
Epoch 2, train accuracy = 40.87
test accuracy = 40.1
Saving..
Total time since training starts:  99.7473

Epoch: 3
Epoch 3, train accuracy = 43.848
test accuracy = 41.95
Saving..
Total time since training starts:  133.5783

Epoch: 4
Epoch 4, train accuracy = 46.38
test accuracy = 46.01
Saving..
Total time since training starts:  165.4129

Epoch: 5
Epoch 5, train accuracy = 48.234
test accuracy = 45.92
Total time since training starts:  197.5919

Epoch: 6
Epoch 6, train accuracy = 49.138
test accuracy = 47.95
Saving..
Total time since training starts:  229.2697

Epoch: 7
Epoch 7, train accuracy = 49.526
test accuracy = 45.86
Total time since training starts:  261.4532

Epoch: 8
Epoch 8, train accuracy = 50.288
test accuracy = 48.15
Saving..
Total time since training starts:  294.5035

Epoch: 9
Epoch 9, train accuracy = 49.834
test accuracy = 47.42
Total time since training starts:  326.5525

Epoch: 10
Epoch 10, train accuracy = 51.102
test accuracy = 50.65
Saving..
Total time since training starts:  361.1368

Epoch: 11
Epoch 11, train accuracy = 51.132
test accuracy = 47.6
Total time since training starts:  393.5255

Epoch: 12
Epoch 12, train accuracy = 51.714
test accuracy = 49.42
Total time since training starts:  428.0052

Epoch: 13
Epoch 13, train accuracy = 51.402
test accuracy = 50.34
Total time since training starts:  461.0134

Epoch: 14
Epoch 14, train accuracy = 52.32
test accuracy = 44.45
Total time since training starts:  496.4982

Epoch: 15
Epoch 15, train accuracy = 52.134
test accuracy = 49.71
Total time since training starts:  528.3949

Epoch: 16
Epoch 16, train accuracy = 52.064
test accuracy = 46.2
Total time since training starts:  561.3334

Epoch: 17
Epoch 17, train accuracy = 52.308
test accuracy = 47.31
Total time since training starts:  595.1076

Epoch: 18
Epoch 18, train accuracy = 52.144
test accuracy = 48.45
Total time since training starts:  625.7886

Epoch: 19
Epoch 19, train accuracy = 48.066
test accuracy = 43.61
Total time since training starts:  658.212

Epoch: 20
Epoch 20, train accuracy = 51.534
test accuracy = 49.13
Total time since training starts:  691.3764

Epoch: 21
Epoch 21, train accuracy = 52.564
test accuracy = 47.03
Total time since training starts:  723.429

Epoch: 22
Epoch 22, train accuracy = 52.346
test accuracy = 48.07
Total time since training starts:  756.2219

Epoch: 23
Epoch 23, train accuracy = 52.714
test accuracy = 46.04
Total time since training starts:  788.156

Epoch: 24
Epoch 24, train accuracy = 52.94
test accuracy = 47.82
Total time since training starts:  820.3065

Epoch: 25
Epoch 25, train accuracy = 52.08
test accuracy = 43.8
Total time since training starts:  853.3862

Epoch: 26
Epoch 26, train accuracy = 53.276
test accuracy = 44.0
Total time since training starts:  884.9288

Epoch: 27
Epoch 27, train accuracy = 51.176
test accuracy = 47.29
Total time since training starts:  918.1041

Epoch: 28
Epoch 28, train accuracy = 49.59
test accuracy = 38.0
Total time since training starts:  951.549

Epoch: 29
Epoch 29, train accuracy = 48.496
test accuracy = 34.84
Total time since training starts:  983.6158

Epoch: 30
Epoch 30, train accuracy = 47.226
test accuracy = 44.52
Total time since training starts:  1016.7748

Epoch: 31
Epoch 31, train accuracy = 47.378
test accuracy = 44.92
Total time since training starts:  1049.2764

Epoch: 32
Epoch 32, train accuracy = 48.682
test accuracy = 41.74
Total time since training starts:  1081.5943

Epoch: 33
Epoch 33, train accuracy = 51.12
test accuracy = 44.7
Total time since training starts:  1115.4649

Epoch: 34
Epoch 34, train accuracy = 51.808
test accuracy = 45.87
Total time since training starts:  1147.9761

Epoch: 35
Epoch 35, train accuracy = 51.12
test accuracy = 47.11
Total time since training starts:  1182.4806

Epoch: 36
Epoch 36, train accuracy = 49.084
test accuracy = 47.81
Total time since training starts:  1214.9345

Epoch: 37
Epoch 37, train accuracy = 50.8
test accuracy = 45.26
Total time since training starts:  1249.6094

Epoch: 38
Epoch 38, train accuracy = 50.798
test accuracy = 47.66
Total time since training starts:  1283.1925

Epoch: 39
Epoch 39, train accuracy = 50.118
test accuracy = 47.0
Total time since training starts:  1316.2362

Epoch: 40
Epoch 40, train accuracy = 49.996
test accuracy = 44.82
Total time since training starts:  1349.2542

Epoch: 41
Epoch 41, train accuracy = 13.176
test accuracy = 10.0
Total time since training starts:  1382.3903

Epoch: 42
Epoch 42, train accuracy = 9.796
test accuracy = 10.0
Total time since training starts:  1413.48

Epoch: 43
Epoch 43, train accuracy = 10.086
test accuracy = 10.0
Total time since training starts:  1443.5033

Epoch: 44
Epoch 44, train accuracy = 9.938
test accuracy = 10.0
Total time since training starts:  1474.6707

Epoch: 45
Epoch 45, train accuracy = 10.016
test accuracy = 10.0
Total time since training starts:  1505.5971

Epoch: 46
Epoch 46, train accuracy = 10.022
test accuracy = 10.0
Total time since training starts:  1536.8498

Epoch: 47
Epoch 47, train accuracy = 10.09
test accuracy = 10.0
Total time since training starts:  1569.2727

Epoch: 48
Epoch 48, train accuracy = 10.042
test accuracy = 10.0
Total time since training starts:  1601.1688

Epoch: 49
Epoch 49, train accuracy = 9.982
test accuracy = 10.0
Total time since training starts:  1633.8003

Epoch: 50
Epoch 50, train accuracy = 9.788
test accuracy = 10.0
Total time since training starts:  1665.226

Epoch: 51
Epoch 51, train accuracy = 9.88
test accuracy = 10.0
Total time since training starts:  1695.3394

Epoch: 52
Epoch 52, train accuracy = 10.092
test accuracy = 10.0
Total time since training starts:  1725.1

Epoch: 53
Epoch 53, train accuracy = 10.036
test accuracy = 10.0
Total time since training starts:  1754.797

Epoch: 54
Epoch 54, train accuracy = 9.928
test accuracy = 10.0
Total time since training starts:  1786.9272

Epoch: 55
Epoch 55, train accuracy = 9.824
test accuracy = 10.0
Total time since training starts:  1819.8183

Epoch: 56
Epoch 56, train accuracy = 9.67
test accuracy = 10.0
Total time since training starts:  1848.945

Epoch: 57
Epoch 57, train accuracy = 9.984
test accuracy = 10.0
Total time since training starts:  1877.6436

Epoch: 58
Epoch 58, train accuracy = 9.966
test accuracy = 10.0
Total time since training starts:  1907.5675

Epoch: 59
Epoch 59, train accuracy = 9.754
test accuracy = 10.0
Total time since training starts:  1938.397

Epoch: 60
Epoch 60, train accuracy = 9.852
test accuracy = 10.0
Total time since training starts:  1968.1221

Epoch: 61
Epoch 61, train accuracy = 9.94
test accuracy = 10.0
Total time since training starts:  1997.4892

Epoch: 62
Epoch 62, train accuracy = 9.886
test accuracy = 10.0
Total time since training starts:  2027.6459

Epoch: 63
Epoch 63, train accuracy = 9.842
test accuracy = 10.0
Total time since training starts:  2057.1195

Epoch: 64
Epoch 64, train accuracy = 9.816
test accuracy = 10.0
Total time since training starts:  2089.8477

Epoch: 65
Epoch 65, train accuracy = 9.63
test accuracy = 10.0
Total time since training starts:  2121.9665

Epoch: 66
Epoch 66, train accuracy = 9.912
test accuracy = 10.0
Total time since training starts:  2154.6075

Epoch: 67
Epoch 67, train accuracy = 9.948
test accuracy = 10.0
Total time since training starts:  2188.3662

Epoch: 68
Epoch 68, train accuracy = 9.944
test accuracy = 10.0
Total time since training starts:  2220.1603

Epoch: 69
Epoch 69, train accuracy = 9.948
test accuracy = 10.0
Total time since training starts:  2252.206

Epoch: 70
Epoch 70, train accuracy = 9.912
test accuracy = 10.0
Total time since training starts:  2285.1222

Epoch: 71
Epoch 71, train accuracy = 10.032
test accuracy = 10.0
Total time since training starts:  2316.0937

Epoch: 72
Epoch 72, train accuracy = 9.926
test accuracy = 10.0
Total time since training starts:  2344.7579

Epoch: 73
Epoch 73, train accuracy = 9.922
test accuracy = 10.0
Total time since training starts:  2375.0283

Epoch: 74
Epoch 74, train accuracy = 9.796
test accuracy = 10.0
Total time since training starts:  2404.8153

Epoch: 75
Epoch 75, train accuracy = 10.006
test accuracy = 10.0
Total time since training starts:  2437.4915

Epoch: 76
Epoch 76, train accuracy = 9.972
test accuracy = 10.0
Total time since training starts:  2468.2409

Epoch: 77
Epoch 77, train accuracy = 9.862
test accuracy = 10.0
Total time since training starts:  2500.422

Epoch: 78
Epoch 78, train accuracy = 9.754
test accuracy = 10.0
Total time since training starts:  2531.8419

Epoch: 79
Epoch 79, train accuracy = 9.948
test accuracy = 10.0
Total time since training starts:  2561.9127

Epoch: 80
Epoch 80, train accuracy = 9.538
test accuracy = 10.0
Total time since training starts:  2593.915

Epoch: 81
Epoch 81, train accuracy = 9.952
test accuracy = 10.0
Total time since training starts:  2623.2846

Epoch: 82
Epoch 82, train accuracy = 9.802
test accuracy = 10.0
Total time since training starts:  2654.0313

Epoch: 83
Epoch 83, train accuracy = 9.922
test accuracy = 10.0
Total time since training starts:  2684.597

Epoch: 84
Epoch 84, train accuracy = 10.044
test accuracy = 10.0
Total time since training starts:  2715.1834

Epoch: 85
Epoch 85, train accuracy = 10.034
test accuracy = 10.0
Total time since training starts:  2744.7562

Epoch: 86
Epoch 86, train accuracy = 9.848
test accuracy = 10.0
Total time since training starts:  2776.911

Epoch: 87
Epoch 87, train accuracy = 9.914
test accuracy = 10.0
Total time since training starts:  2806.6105

Epoch: 88
Epoch 88, train accuracy = 10.18
test accuracy = 10.0
Total time since training starts:  2836.9194

Epoch: 89
Epoch 89, train accuracy = 10.238
test accuracy = 10.0
Total time since training starts:  2867.0632

Epoch: 90
Epoch 90, train accuracy = 9.882
test accuracy = 10.0
Total time since training starts:  2898.3037

Epoch: 91
Epoch 91, train accuracy = 10.162
test accuracy = 10.0
Total time since training starts:  2929.8822

Epoch: 92
Epoch 92, train accuracy = 9.798
test accuracy = 10.0
Total time since training starts:  2958.6602

Epoch: 93
Epoch 93, train accuracy = 9.89
test accuracy = 10.0
Total time since training starts:  2987.8363

Epoch: 94
Epoch 94, train accuracy = 10.08
test accuracy = 10.0
Total time since training starts:  3017.446

Epoch: 95
Epoch 95, train accuracy = 9.862
test accuracy = 10.0
Total time since training starts:  3046.6737

Epoch: 96
Epoch 96, train accuracy = 9.87
test accuracy = 10.0
Total time since training starts:  3077.2568

Epoch: 97
Epoch 97, train accuracy = 9.946
test accuracy = 10.0
Total time since training starts:  3110.5288

Epoch: 98
Epoch 98, train accuracy = 9.894
test accuracy = 10.0
Total time since training starts:  3142.9664

Epoch: 99
Epoch 99, train accuracy = 10.222
test accuracy = 10.0
Total time since training starts:  3174.9209

Epoch: 100
Epoch 100, train accuracy = 9.952
test accuracy = 10.0
Total time since training starts:  3205.9333

Epoch: 101
Epoch 101, train accuracy = 10.052
test accuracy = 10.0
Total time since training starts:  3237.6111

Epoch: 102
Epoch 102, train accuracy = 9.626
test accuracy = 10.0
Total time since training starts:  3268.3611

Epoch: 103
Epoch 103, train accuracy = 9.944
test accuracy = 10.0
Total time since training starts:  3299.8763

Epoch: 104
Epoch 104, train accuracy = 9.808
test accuracy = 10.0
Total time since training starts:  3330.6406

Epoch: 105
Epoch 105, train accuracy = 9.926
test accuracy = 10.0
Total time since training starts:  3364.1834

Epoch: 106
Epoch 106, train accuracy = 9.886
test accuracy = 10.0
Total time since training starts:  3396.7027

Epoch: 107
Epoch 107, train accuracy = 9.988
test accuracy = 10.0
Total time since training starts:  3428.883

Epoch: 108
Epoch 108, train accuracy = 9.926
test accuracy = 10.0
Total time since training starts:  3459.2114

Epoch: 109
Epoch 109, train accuracy = 9.91
test accuracy = 10.0
Total time since training starts:  3490.8828

Epoch: 110
Epoch 110, train accuracy = 9.846
test accuracy = 10.0
Total time since training starts:  3519.711

Epoch: 111
Epoch 111, train accuracy = 9.882
test accuracy = 10.0
Total time since training starts:  3550.0263

Epoch: 112
Epoch 112, train accuracy = 9.79
test accuracy = 10.0
Total time since training starts:  3580.7512

Epoch: 113
Epoch 113, train accuracy = 9.712
test accuracy = 10.0
Total time since training starts:  3612.0305

Epoch: 114
Epoch 114, train accuracy = 10.17
test accuracy = 10.0
Total time since training starts:  3643.0235

Epoch: 115
Epoch 115, train accuracy = 9.936
test accuracy = 10.0
Total time since training starts:  3675.0733

Epoch: 116
Epoch 116, train accuracy = 9.832
test accuracy = 10.0
Total time since training starts:  3704.2879

Epoch: 117
Epoch 117, train accuracy = 9.806
test accuracy = 10.0
Total time since training starts:  3735.0137

Epoch: 118
Epoch 118, train accuracy = 9.826
test accuracy = 10.0
Total time since training starts:  3767.9628

Epoch: 119
Epoch 119, train accuracy = 10.22
test accuracy = 10.0
Total time since training starts:  3797.7143

Epoch: 120
Epoch 120, train accuracy = 9.958
test accuracy = 10.0
Total time since training starts:  3830.7932

Epoch: 121
Epoch 121, train accuracy = 9.828
test accuracy = 10.0
Total time since training starts:  3862.2439

Epoch: 122
Epoch 122, train accuracy = 9.77
test accuracy = 10.0
Total time since training starts:  3892.7143

Epoch: 123
Epoch 123, train accuracy = 10.048
test accuracy = 10.0
Total time since training starts:  3922.0518

Epoch: 124
Epoch 124, train accuracy = 9.866
test accuracy = 10.0
Total time since training starts:  3952.1254

Epoch: 125
Epoch 125, train accuracy = 9.786
test accuracy = 10.0
Total time since training starts:  3982.2381

Epoch: 126
Epoch 126, train accuracy = 9.864
test accuracy = 10.0
Total time since training starts:  4012.4614

Epoch: 127
Epoch 127, train accuracy = 9.97
test accuracy = 10.0
Total time since training starts:  4043.0498

Epoch: 128
Epoch 128, train accuracy = 9.992
test accuracy = 10.0
Total time since training starts:  4073.9777

Epoch: 129
Epoch 129, train accuracy = 9.704
test accuracy = 10.0
Total time since training starts:  4105.1468

Epoch: 130
Epoch 130, train accuracy = 10.004
test accuracy = 10.0
Total time since training starts:  4136.4385

Epoch: 131
Epoch 131, train accuracy = 9.872
test accuracy = 10.0
Total time since training starts:  4165.817

Epoch: 132
Epoch 132, train accuracy = 10.22
test accuracy = 10.0
Total time since training starts:  4194.8796

Epoch: 133
Epoch 133, train accuracy = 10.068
test accuracy = 10.0
Total time since training starts:  4225.083

Epoch: 134
Epoch 134, train accuracy = 9.938
test accuracy = 10.0
Total time since training starts:  4254.0966

Epoch: 135
Epoch 135, train accuracy = 10.054
test accuracy = 10.0
Total time since training starts:  4284.9899

Epoch: 136
Epoch 136, train accuracy = 10.164
test accuracy = 10.0
Total time since training starts:  4315.2376

Epoch: 137
Epoch 137, train accuracy = 9.708
test accuracy = 10.0
Total time since training starts:  4344.4358

Epoch: 138
Epoch 138, train accuracy = 9.784
test accuracy = 10.0
Total time since training starts:  4374.8351

Epoch: 139
Epoch 139, train accuracy = 9.91
test accuracy = 10.0
Total time since training starts:  4405.9003

Epoch: 140
Epoch 140, train accuracy = 10.09
test accuracy = 10.0
Total time since training starts:  4435.8614

Epoch: 141
Epoch 141, train accuracy = 10.162
test accuracy = 10.0
Total time since training starts:  4468.0774

Epoch: 142
Epoch 142, train accuracy = 9.958
test accuracy = 10.0
Total time since training starts:  4498.1542

Epoch: 143
Epoch 143, train accuracy = 9.934
test accuracy = 10.0
Total time since training starts:  4529.7787

Epoch: 144
Epoch 144, train accuracy = 9.816
test accuracy = 10.0
Total time since training starts:  4559.0166

Epoch: 145
Epoch 145, train accuracy = 10.112
test accuracy = 10.0
Total time since training starts:  4590.6292

Epoch: 146
Epoch 146, train accuracy = 9.976
test accuracy = 10.0
Total time since training starts:  4620.9043

Epoch: 147
Epoch 147, train accuracy = 9.788
test accuracy = 10.0
Total time since training starts:  4652.0715

Epoch: 148
Epoch 148, train accuracy = 9.858
test accuracy = 10.0
Total time since training starts:  4683.0518

Epoch: 149
Epoch 149, train accuracy = 10.196
test accuracy = 10.0
Total time since training starts:  4713.0018

Epoch: 150
Epoch 150, train accuracy = 9.91
test accuracy = 10.0
Total time since training starts:  4742.0467

Epoch: 151
Epoch 151, train accuracy = 10.1
test accuracy = 10.0
Total time since training starts:  4772.3435

Epoch: 152
Epoch 152, train accuracy = 10.092
test accuracy = 10.0
Total time since training starts:  4803.4862

Epoch: 153
Epoch 153, train accuracy = 9.99
test accuracy = 10.0
Total time since training starts:  4833.0148

Epoch: 154
Epoch 154, train accuracy = 9.736
test accuracy = 10.0
Total time since training starts:  4864.6953

Epoch: 155
Epoch 155, train accuracy = 10.182
test accuracy = 10.0
Total time since training starts:  4894.2963

Epoch: 156
Epoch 156, train accuracy = 9.894
test accuracy = 10.0
Total time since training starts:  4923.7006

Epoch: 157
Epoch 157, train accuracy = 9.978
test accuracy = 10.0
Total time since training starts:  4955.2869

Epoch: 158
Epoch 158, train accuracy = 9.86
test accuracy = 10.0
Total time since training starts:  4984.7344

Epoch: 159
Epoch 159, train accuracy = 9.982
test accuracy = 10.0
Total time since training starts:  5014.7552

Epoch: 160
Epoch 160, train accuracy = 10.126
test accuracy = 10.0
Total time since training starts:  5045.9174

Epoch: 161
Epoch 161, train accuracy = 9.888
test accuracy = 10.0
Total time since training starts:  5078.9551

Epoch: 162
Epoch 162, train accuracy = 10.032
test accuracy = 10.0
Total time since training starts:  5109.809

Epoch: 163
Epoch 163, train accuracy = 9.74
test accuracy = 10.0
Total time since training starts:  5139.7321

Epoch: 164
Epoch 164, train accuracy = 9.824
test accuracy = 10.0
Total time since training starts:  5169.7659

Epoch: 165
Epoch 165, train accuracy = 9.836
test accuracy = 10.0
Total time since training starts:  5200.8913

Epoch: 166
Epoch 166, train accuracy = 9.936
test accuracy = 10.0
Total time since training starts:  5230.5993

Epoch: 167
Epoch 167, train accuracy = 10.034
test accuracy = 10.0
Total time since training starts:  5260.3071

Epoch: 168
Epoch 168, train accuracy = 9.936
test accuracy = 10.0
Total time since training starts:  5291.5095

Epoch: 169
Epoch 169, train accuracy = 9.93
test accuracy = 10.0
Total time since training starts:  5322.5398

Epoch: 170
Epoch 170, train accuracy = 9.984
test accuracy = 10.0
Total time since training starts:  5352.1113

Epoch: 171
Epoch 171, train accuracy = 10.284
test accuracy = 10.0
Total time since training starts:  5380.6605

Epoch: 172
Epoch 172, train accuracy = 10.04
test accuracy = 10.0
Total time since training starts:  5410.7313

Epoch: 173
Epoch 173, train accuracy = 10.126
test accuracy = 10.0
Total time since training starts:  5440.8326

Epoch: 174
Epoch 174, train accuracy = 9.746
test accuracy = 10.0
Total time since training starts:  5472.0756

Epoch: 175
Epoch 175, train accuracy = 9.746
test accuracy = 10.0
Total time since training starts:  5503.0737

Epoch: 176
Epoch 176, train accuracy = 10.05
test accuracy = 10.0
Total time since training starts:  5535.9194

Epoch: 177
Epoch 177, train accuracy = 10.086
test accuracy = 10.0
Total time since training starts:  5568.2434

Epoch: 178
Epoch 178, train accuracy = 9.788
test accuracy = 10.0
Total time since training starts:  5599.5695

Epoch: 179
Epoch 179, train accuracy = 10.096
test accuracy = 10.0
Total time since training starts:  5631.1565

Epoch: 180
Epoch 180, train accuracy = 10.124
test accuracy = 10.0
Total time since training starts:  5664.3892

Epoch: 181
Epoch 181, train accuracy = 9.718
test accuracy = 10.0
Total time since training starts:  5696.6571

Epoch: 182
Epoch 182, train accuracy = 10.002
test accuracy = 10.0
Total time since training starts:  5729.5978

Epoch: 183
Epoch 183, train accuracy = 10.004
test accuracy = 10.0
Total time since training starts:  5764.1232

Epoch: 184
Epoch 184, train accuracy = 9.836
test accuracy = 10.0
Total time since training starts:  5796.1852

Epoch: 185
Epoch 185, train accuracy = 9.916
test accuracy = 10.0
Total time since training starts:  5828.1373

Epoch: 186
Epoch 186, train accuracy = 10.066
test accuracy = 10.0
Total time since training starts:  5862.514

Epoch: 187
Epoch 187, train accuracy = 10.106
test accuracy = 10.0
Total time since training starts:  5894.7975

Epoch: 188
Epoch 188, train accuracy = 10.004
test accuracy = 10.0
Total time since training starts:  5925.5791

Epoch: 189
Epoch 189, train accuracy = 10.332
test accuracy = 10.0
Total time since training starts:  5961.2575

Epoch: 190
Epoch 190, train accuracy = 9.866
test accuracy = 10.0
Total time since training starts:  5991.9039

Epoch: 191
Epoch 191, train accuracy = 10.078
test accuracy = 10.0
Total time since training starts:  6024.4944

Epoch: 192
Epoch 192, train accuracy = 9.88
test accuracy = 10.0
Total time since training starts:  6056.4918

Epoch: 193
Epoch 193, train accuracy = 9.846
test accuracy = 10.0
Total time since training starts:  6089.637

Epoch: 194
Epoch 194, train accuracy = 10.204
test accuracy = 10.0
Total time since training starts:  6120.9242

Epoch: 195
Epoch 195, train accuracy = 9.752
test accuracy = 10.0
Total time since training starts:  6154.0019

Epoch: 196
Epoch 196, train accuracy = 9.798
test accuracy = 10.0
Total time since training starts:  6185.3368

Epoch: 197
Epoch 197, train accuracy = 10.192
test accuracy = 10.0
Total time since training starts:  6220.2835

Epoch: 198
Epoch 198, train accuracy = 10.022
test accuracy = 10.0
Total time since training starts:  6251.495

Epoch: 199
Epoch 199, train accuracy = 9.818
test accuracy = 10.0
Total time since training starts:  6283.937
"""

line_num = 0

train_accuracy = []
test_accuracy  = []
times          = []

for line in input.split("\n"):
    metrics = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if "train accuracy" in line:
        train_acc = float(metrics[1])
        train_accuracy.append(train_acc)
    elif "test accuracy" in line:
        test_acc = float(metrics[0])
        test_accuracy.append(test_acc)
    elif "time since" in line:
        time      = int(float(metrics[0]))
        times.append(time)

    if len(times) > 40:
        break

print(train_accuracy)


plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.legend(["train accuracy", "test accuracy"])
plt.ylim([0, 100])
plt.xlabel("Epochs")

# xticks = np.arange(0, 1500, 200)
# plt.xticks(xticks)

plt.show()
