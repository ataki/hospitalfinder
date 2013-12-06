# Cross validation error over m

Note: if we predict everything using the mean of Y, we get an average error of 7.95366.

## Linear regression

```
m = 50
avg training error:  6.26825011649e-12
avg test error:  56.391397196
m = 100
avg training error:  6.22177446433
avg test error:  12.3924700647
m = 200
avg training error:  6.80495151093
avg test error:  9.02893048353
m = 400
avg training error:  6.4848471984
avg test error:  7.45073906012
m = 600
avg training error:  6.79505469551
avg test error:  7.51733251002
m = 800
avg training error:  6.91523970364
avg test error:  7.39105364834
m = 1000
avg training error:  6.95807992327
avg test error:  7.36557636238
m = 1200
avg training error:  6.86788197268
avg test error:  7.13909233789
m = 1400
avg training error:  6.94272501507
avg test error:  7.21924156886
m = 1600
avg training error:  6.9068926673
avg test error:  7.10953415585
m = 1800
avg training error:  6.93582695281
avg test error:  7.12947755264
m = 2000
avg training error:  7.05929192192
avg test error:  7.22537363991
```

## Lasso regression

```
alpha = 0.01

intercept = 20.6961249257
coef =
[ -1.50200998e-02   4.61625208e-04  -3.47476362e-02  -3.58035163e-01
  -5.00702771e-01   5.59371539e-01   8.32831561e-02  -5.77106949e-01
   6.72632189e-02  -0.00000000e+00   8.25886095e-01   0.00000000e+00
  -5.49561656e-03  -5.02678124e-01  -1.04553775e-01  -0.00000000e+00
  -8.85434701e-03   7.01621453e-01  -9.63069188e-02  -1.37801344e-03
  -0.00000000e+00   0.00000000e+00   1.11266547e+00  -2.72828008e-01
   6.81360794e-01  -6.85217122e-01   0.00000000e+00  -2.65256993e-01
   4.57742762e-01  -0.00000000e+00   3.57241483e+00  -1.89365513e+00
  -2.49674549e-05   3.07503847e-01  -1.51929369e+00  -3.76101606e-01
  -4.57573210e-01  -1.09830620e+00   7.54042136e-01   1.33878915e+00
  -1.34271305e+00   0.00000000e+00  -2.14673608e+00  -1.47150713e+00
   2.04780225e+00  -1.59853527e-01   9.99699971e-01   0.00000000e+00
  -1.52366763e+00   3.51121692e-01  -3.51546034e-01   1.29764899e-01
   7.17316377e-01]

selected features =
[False False False  True  True  True False  True False False  True False
 False  True False False False False False False False False  True False
  True  True False  True  True False  True  True False  True  True  True
  True  True  True  True  True False  True  True  True False  True False
  True  True  True False  True]
feature rankings =
[10 15 11  1  1  1  8  1  9 23  1 24 12  1  7 17 13  5  6 14 21 18  1  2  1
  1 25  1  1 22  1  1 16  1  1  1  1  1  1  1  1 19  1  1  1  3  1 20  1  1
  1  4  1]

m = 10
avg training error:  0.0285124246708
avg test error:  14.4359472611

m = 20
avg training error:  0.108409374348
avg test error:  14.2138163404

m = 50
avg training error:  4.53839202549
avg test error:  11.8294212687

m = 100
avg training error:  7.04948811831
avg test error:  10.6319997163

m = 200
avg training error:  7.06390883624
avg test error:  8.59087781633

m = 400
avg training error:  7.04205911026
avg test error:  7.68428479231

m = 600
avg training error:  6.92979451778
avg test error:  7.29214041786

m = 800
avg training error:  7.0905717212
avg test error:  7.35378509931

m = 1000
avg training error:  6.92595261445
avg test error:  7.1460242202

m = 1200
avg training error:  6.87785737848
avg test error:  7.06683290793

m = 1400
avg training error:  7.10671039961
avg test error:  7.26513692779

m = 1600
avg training error:  7.1314451893
avg test error:  7.26312505971

m = 1800
avg training error:  7.17451352891
avg test error:  7.29679796787

m = 2000
avg training error:  7.12997501585
avg test error:  7.24417845959

m = 2184
avg training error:  7.10657442154
avg test error:  7.20810095153
```

## Ridge regression

Select alpha with CV.

```
alpha = 0.1

intercept = 14.584214951
coef =
[ -1.43783598e-02   1.29018881e-04   6.58077568e-04  -3.14615215e-01
  -5.60324737e-01   5.05968376e-01   8.23793001e-02  -6.84686897e-01
   3.05573983e-01   1.13536524e-02   9.42089724e-01   0.00000000e+00
  -3.65764336e-03  -5.42004792e-01  -1.24062604e-02  -1.70494669e-01
  -4.61954878e-04   1.46533729e-02  -2.90005231e-03   2.96556208e-03
   9.48985015e-02   0.00000000e+00   1.00494345e+00  -1.32962684e-01
   6.43576400e-01  -4.87775594e-01   2.86503226e+00  -7.07665445e-02
   2.46947904e-01  -4.26573849e-02   3.23128092e+00  -1.82611126e+00
  -1.64361312e-05   2.29883509e-01  -1.43713608e+00  -5.58358517e-01
  -3.90910559e-01  -1.07545389e+00   8.13852775e-01   1.13863918e+00
  -1.18201337e+00  -2.99374173e-02  -1.74931355e+00  -1.30377157e+00
   1.78910534e+00  -1.95332758e-01   7.08702871e-01   0.00000000e+00
  -1.33841794e+00   3.06545279e-01  -6.08646505e-01   7.92718873e-02
   6.81251439e-01]

selected features =
[False False False  True  True  True False  True  True False  True False
 False  True False  True False False False False  True False  True  True
  True  True  True False  True False  True  True False  True  True  True
  True  True  True  True  True False  True  True  True  True  True False
  True  True  True False  True]
feature rankings =
[ 7 16 11  1  1  1  2  1  1  9  1 18 13  1  8  1 15 10 12 14  1 19  1  1  1
  1  1  4  1  5  1  1 17  1  1  1  1  1  1  1  1  6  1  1  1  1  1 20  1  1
  1  3  1]

m = 10
avg training error:  0.0040250003456
avg test error:  10.1855177955

m = 20
avg training error:  0.0138387262792
avg test error:  9.90722731455

m = 50
avg training error:  8.61269625946
avg test error:  9.75399896174

m = 100
avg training error:  7.23753751783
avg test error:  9.35165443509

m = 200
avg training error:  6.9391113515
avg test error:  8.22657039132

m = 400
avg training error:  6.82697831272
avg test error:  7.50248131118

m = 600
avg training error:  6.74225377927
avg test error:  7.09053175945

m = 800
avg training error:  6.8727895153
avg test error:  7.14487570958

m = 1000
avg training error:  7.07553311683
avg test error:  7.29187825442

m = 1200
avg training error:  6.98405153423
avg test error:  7.17321090155

m = 1400
avg training error:  7.04302360979
avg test error:  7.21040723882

m = 1600
avg training error:  7.18600635323
avg test error:  7.34803750556

m = 1800
avg training error:  7.05037212767
avg test error:  7.2096947513

m = 2000
avg training error:  7.07803358334
avg test error:  7.19660260872

m = 2184
avg training error:  7.11529554578
avg test error:  7.22242722289
```

## Elastic net

```
alpha = 0.1
l1_ratio = 0.1

intercept =
8.34766774815
coef =
[ -2.62005939e-02   4.46742031e-04  -2.92191072e-02  -1.96616082e-01
  -2.11778938e-01   4.28997730e-01   9.53503721e-02  -2.77903353e-01
   0.00000000e+00  -1.53584366e-01   5.19517107e-01   0.00000000e+00
  -5.94240061e-03  -3.77221769e-01  -1.99975854e-02   0.00000000e+00
  -7.83171476e-03   1.68710530e-01   9.56694553e-03   9.17868062e-03
   0.00000000e+00   0.00000000e+00   1.09631348e+00  -1.63923984e-01
   6.57022644e-01  -2.66391957e-01   0.00000000e+00  -1.46058343e-01
   3.23107378e-01  -4.95034650e-02   1.72134497e+00  -1.14243798e+00
  -4.34156211e-05   2.46695374e-01  -1.15532983e+00  -5.56566430e-01
  -4.14560602e-01  -8.53494859e-01   5.18069136e-01   7.60700022e-01
  -1.04457789e+00  -0.00000000e+00  -1.08445770e+00  -1.08682828e+00
   1.67941270e+00  -1.62592828e-01   6.78903627e-02   0.00000000e+00
  -1.36724376e+00   3.40798915e-01  -1.98049902e-01   9.85399378e-02
   6.15682932e-01]

selected features =
[False False False  True  True  True  True  True False  True  True False
 False  True False False False False False False False False  True  True
  True  True False  True  True False  True  True False  True  True  True
  True  True  True  True  True False  True  True  True  True  True False
  True  True  True False  True]
feature ranking =
[ 5 12  7  1  1  1  1  1 17  1  1 15  9  1  4 14 10  6 11  8 18 19  1  1  1
  1 21  1  1  3  1  1 13  1  1  1  1  1  1  1  1 20  1  1  1  1  1 16  1  1
  1  2  1]

m = 10
avg training error:  0.436347415561
avg test error:  7.95981723298

m = 20
avg training error:  2.1457496262
avg test error:  13.0911689908

m = 50
avg training error:  4.70638927413
avg test error:  8.52973282149

m = 100
avg training error:  6.15547014818
avg test error:  8.01376225381

m = 200
avg training error:  6.83355547181
avg test error:  7.82795139967

m = 400
avg training error:  6.82309981775
avg test error:  7.28870954998

m = 600
avg training error:  7.23004618132
avg test error:  7.55137897304

m = 800
avg training error:  7.13693048091
avg test error:  7.36610338272

m = 1000
avg training error:  7.25766003398
avg test error:  7.43322982337

m = 1200
avg training error:  7.21179174131
avg test error:  7.35402393215

m = 1400
avg training error:  7.14676648808
avg test error:  7.2770297822

m = 1600
avg training error:  7.20393082617
avg test error:  7.31742451215

m = 1800
avg training error:  7.09040242521
avg test error:  7.19743974927

m = 2000
avg training error:  7.11618956626
avg test error:  7.20177278853

m = 2184
avg training error:  7.15732085645
avg test error:  7.24499927829
```
