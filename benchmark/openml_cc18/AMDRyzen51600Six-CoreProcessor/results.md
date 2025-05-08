
## Graphs
 All times are specified in seconds
![alt](train_accuracy.png)

![alt](train_time.png)

![alt](test_time.png)

![alt](samples_train_time.png)

![alt](samples_test_time.png)

![alt](samples_speedup_train_time.png)

![alt](samples_speedup_test_time.png)

![alt](features_train_time.png)

![alt](features_test_time.png)

![alt](features_speedup_train_time.png)

![alt](features_speedup_test_time.png)


# Benchmark table
| model                      | dataset                                |   train_accuracy |    train_time |   test_time |   samples |   features |
|:---------------------------|:---------------------------------------|-----------------:|--------------:|------------:|----------:|-----------:|
| sklearn.tree               | kr-vs-kp                               |        0.940864  |   0.00493156  | 0.00300394  |      3196 |         36 |
| sklearn.tree               | letter                                 |        0.60975   |   0.0094374   | 0.00189309  |     20000 |         16 |
| sklearn.tree               | balance-scale                          |        0.7984    |   0.000575043 | 0.000299199 |       625 |          4 |
| sklearn.tree               | mfeat-factors                          |        0.8185    |   0.00768132  | 0.000363372 |      2000 |         32 |
| sklearn.tree               | mfeat-fourier                          |        0.785     |   0.00733189  | 0.000427128 |      2000 |         32 |
| sklearn.tree               | breast-w                               |        0.942775  |   0.000544134 | 0.000257525 |       699 |          9 |
| sklearn.tree               | mfeat-karhunen                         |        0.8495    |   0.00811382  | 0.000371799 |      2000 |         32 |
| sklearn.tree               | mfeat-morphological                    |        0.707     |   0.000915246 | 0.000317751 |      2000 |          6 |
| sklearn.tree               | mfeat-zernike                          |        0.711     |   0.00770823  | 0.000504599 |      2000 |         32 |
| sklearn.tree               | cmc                                    |        0.473184  |   0.000969205 | 0.000615932 |      1473 |          9 |
| sklearn.tree               | optdigits                              |        0.837011  |   0.0192989   | 0.000489309 |      5620 |         32 |
| sklearn.tree               | credit-approval                        |        0.855072  |   0.00131575  | 0.000761789 |       690 |         15 |
| sklearn.tree               | credit-g                               |        0.7       |   0.00188027  | 0.00110425  |      1000 |         20 |
| sklearn.tree               | pendigits                              |        0.866357  |   0.00591781  | 0.000507916 |     10992 |         16 |
| sklearn.tree               | diabetes                               |        0.735677  |   0.000581013 | 0.000257911 |       768 |          8 |
| sklearn.tree               | spambase                               |        0.81178   |   0.00846284  | 0.000434451 |      4601 |         32 |
| sklearn.tree               | splice                                 |        0.897179  |   0.00799266  | 0.00412268  |      3190 |         60 |
| sklearn.tree               | tic-tac-toe                            |        0.699374  |   0.000809202 | 0.000515452 |       958 |          9 |
| sklearn.tree               | vehicle                                |        0.6513    |   0.00176043  | 0.000814694 |       846 |         18 |
| sklearn.tree               | electricity                            |        0.757393  |   0.00912589  | 0.00165178  |     45312 |          8 |
| sklearn.tree               | satimage                               |        0.833593  |   0.0170945   | 0.000639284 |      6430 |         32 |
| sklearn.tree               | eucalyptus                             |        0.649457  |   0.00137875  | 0.000674697 |       736 |         19 |
| sklearn.tree               | sick                                   |        0.965536  |   0.0035408   | 0.0020442   |      3772 |         29 |
| sklearn.tree               | vowel                                  |        0.753535  |   0.00158869  | 0.000451254 |       990 |         12 |
| sklearn.tree               | isolet                                 |        0.706297  |   0.0349843   | 0.000580085 |      7797 |         32 |
| sklearn.tree               | analcatdata_authorship                 |        0.946492  |   0.00188661  | 0.000309451 |       841 |         32 |
| sklearn.tree               | analcatdata_dmft                       |        0.194479  |   0.000496914 | 0.000317193 |       797 |          4 |
| sklearn.tree               | mnist_784                              |        0.681357  |   0.305256    | 0.00315025  |     70000 |         32 |
| sklearn.tree               | pc4                                    |        0.877915  |   0.00198968  | 0.000327046 |      1458 |         32 |
| sklearn.tree               | pc3                                    |        0.897633  |   0.00185213  | 0.000399359 |      1563 |         32 |
| sklearn.tree               | jm1                                    |        0.806523  |   0.00472139  | 0.000785381 |     10885 |         21 |
| sklearn.tree               | kc2                                    |        0.850575  |   0.00115885  | 0.000542151 |       522 |         21 |
| sklearn.tree               | kc1                                    |        0.845424  |   0.00154958  | 0.000650367 |      2109 |         21 |
| sklearn.tree               | pc1                                    |        0.930568  |   0.00113562  | 0.000588958 |      1109 |         21 |
| sklearn.tree               | adult                                  |        0.760718  |   0.0184432   | 0.011112    |     48842 |         14 |
| sklearn.tree               | Bioresponse                            |        0.623567  |   0.00646343  | 0.000461626 |      3751 |         32 |
| sklearn.tree               | wdbc                                   |        0.940246  |   0.001169    | 0.000333584 |       569 |         30 |
| sklearn.tree               | phoneme                                |        0.754626  |   0.00150276  | 0.000343541 |      5404 |          5 |
| sklearn.tree               | qsar-biodeg                            |        0.776303  |   0.00214816  | 0.000350213 |      1055 |         32 |
| sklearn.tree               | wall-robot-navigation                  |        0.969758  |   0.00682716  | 0.000431827 |      5456 |         24 |
| sklearn.tree               | semeion                                |        0.766478  |   0.00905957  | 0.00033549  |      1593 |         32 |
| sklearn.tree               | ilpd                                   |        0.713551  |   0.000841413 | 0.00049533  |       583 |         10 |
| sklearn.tree               | madelon                                |        0.5       |   0.00436917  | 0.000461519 |      2600 |         32 |
| sklearn.tree               | nomao                                  |        0.892761  |   0.0988794   | 0.0190471   |     34465 |         61 |
| sklearn.tree               | ozone-level-8hr                        |        0.936859  |   0.00288307  | 0.000450212 |      2534 |         32 |
| sklearn.tree               | cnae-9                                 |        0.823148  |   0.0066333   | 0.000455638 |      1080 |         32 |
| sklearn.tree               | first-order-theorem-proving            |        0.429552  |   0.0109696   | 0.000597719 |      6118 |         32 |
| sklearn.tree               | banknote-authentication                |        0.91691   |   0.000827908 | 0.000360721 |      1372 |          4 |
| sklearn.tree               | blood-transfusion-service-center       |        0.762032  |   0.000943043 | 0.000563594 |       748 |          4 |
| sklearn.tree               | PhishingWebsites                       |        0.888919  |   0.00840965  | 0.00576216  |     11055 |         30 |
| sklearn.tree               | cylinder-bands                         |        0.707407  |   0.00204057  | 0.00111676  |       540 |         37 |
| sklearn.tree               | bank-marketing                         |        0.883015  |   0.0168485   | 0.0092257   |     45211 |         16 |
| sklearn.tree               | GesturePhaseSegmentationProcessed      |        0.444343  |   0.0187319   | 0.0006348   |      9873 |         32 |
| sklearn.tree               | har                                    |        0.790659  |   0.0266234   | 0.000655419 |     10299 |         32 |
| sklearn.tree               | dresses-sales                          |        0.58      |   0.00129461  | 0.000882575 |       500 |         12 |
| sklearn.tree               | texture                                |        0.852545  |   0.0208236   | 0.00047696  |      5500 |         32 |
| sklearn.tree               | connect-4                              |        0.658303  |   0.0636795   | 0.0486303   |     67557 |         42 |
| sklearn.tree               | MiceProtein                            |        0.87037   |   0.00417274  | 0.000392406 |      1080 |         32 |
| sklearn.tree               | steel-plates-fault                     |        0.704276  |   0.00420728  | 0.000706189 |      1941 |         27 |
| sklearn.tree               | climate-model-simulation-crashes       |        0.914815  |   0.000661035 | 0.000281025 |       540 |         18 |
| sklearn.tree               | wilt                                   |        0.946063  |   0.00101431  | 0.000295266 |      4839 |          5 |
| sklearn.tree               | car                                    |        0.805556  |   0.000754991 | 0.000490176 |      1728 |          6 |
| sklearn.tree               | segment                                |        0.887879  |   0.00212265  | 0.000317427 |      2310 |         16 |
| sklearn.tree               | mfeat-pixel                            |        0.88      |   0.00926018  | 0.000363759 |      2000 |         32 |
| sklearn.tree               | Fashion-MNIST                          |        0.667943  |   0.271144    | 0.00318968  |     70000 |         32 |
| sklearn.tree               | jungle_chess_2pcs_raw_endgame_complete |        0.6491    |   0.0041905   | 0.00126597  |     44819 |          6 |
| sklearn.tree               | numerai28.6                            |        0.50517   |   0.0378668   | 0.00258362  |     96320 |         21 |
| sklearn.tree               | Devnagari-Script                       |        0.229402  |   0.594973    | 0.00527931  |     92000 |         32 |
| sklearn.tree               | CIFAR_10                               |        0.177533  |   0.11966     | 0.00255593  |     60000 |         32 |
| sklearn.tree               | Internet-Advertisements                |        0.936566  |   0.174965    | 0.117385    |      3279 |       1558 |
| sklearn.tree               | dna                                    |        0.853107  |   0.0203288   | 0.0128005   |      3186 |        180 |
| sklearn.tree               | churn                                  |        0.8706    |   0.00344516  | 0.00147404  |      5000 |         20 |
| sklearnmodels.tree[pandas] | kr-vs-kp                               |        0.942428  |   0.0527716   | 0.0199487   |      3196 |         36 |
| sklearnmodels.tree[pandas] | letter                                 |        0.6089    |   0.428225    | 0.139504    |     20000 |         16 |
| sklearnmodels.tree[pandas] | balance-scale                          |        0.7536    |   0.0100565   | 0.00330457  |       625 |          4 |
| sklearnmodels.tree[pandas] | mfeat-factors                          |        0.838     |   1.02659     | 0.0168648   |      2000 |         32 |
| sklearnmodels.tree[pandas] | mfeat-fourier                          |        0.816     |   1.09505     | 0.0133844   |      2000 |         32 |
| sklearnmodels.tree[pandas] | breast-w                               |        0.958512  |   0.0286312   | 0.00325195  |       699 |          9 |
| sklearnmodels.tree[pandas] | mfeat-karhunen                         |        0.8525    |   0.955033    | 0.0130638   |      2000 |         32 |
| sklearnmodels.tree[pandas] | mfeat-morphological                    |        0.676     |   0.0140692   | 0.0107629   |      2000 |          6 |
| sklearnmodels.tree[pandas] | mfeat-zernike                          |        0.74      |   1.11607     | 0.0133721   |      2000 |         32 |
| sklearnmodels.tree[pandas] | cmc                                    |        0.488798  |   0.0165353   | 0.00886184  |      1473 |          9 |
| sklearnmodels.tree[pandas] | optdigits                              |        0.844128  |   1.04014     | 0.0363983   |      5620 |         32 |
| sklearnmodels.tree[pandas] | credit-approval                        |        0.555072  |   0.0314995   | 0.00300796  |       690 |         15 |
| sklearnmodels.tree[pandas] | credit-g                               |        0.7       |   0.0355607   | 0.00452653  |      1000 |         20 |
| sklearnmodels.tree[pandas] | pendigits                              |        0.858988  |   0.289777    | 0.0706802   |     10992 |         16 |
| sklearnmodels.tree[pandas] | diabetes                               |        0.785156  |   0.034029    | 0.00432091  |       768 |          8 |
| sklearnmodels.tree[pandas] | spambase                               |        0.845251  |   0.375016    | 0.0255598   |      4601 |         32 |
| sklearnmodels.tree[pandas] | splice                                 |        0.797179  |   0.302436    | 0.022736    |      3190 |         60 |
| sklearnmodels.tree[pandas] | tic-tac-toe                            |        0.699374  |   0.00824251  | 0.00517947  |       958 |          9 |
| sklearnmodels.tree[pandas] | vehicle                                |        0.749409  |   0.188561    | 0.00631872  |       846 |         18 |
| sklearnmodels.tree[pandas] | electricity                            |        0.575455  |   3.91781     | 0.183348    |     45312 |          8 |
| sklearnmodels.tree[pandas] | satimage                               |        0.852877  |   0.804853    | 0.0393021   |      6430 |         32 |
| sklearnmodels.tree[pandas] | eucalyptus                             |        0.290761  |   0.0443841   | 0.00322565  |       736 |         19 |
| sklearnmodels.tree[pandas] | sick                                   |        0.938759  |   0.159994    | 0.0273612   |      3772 |         29 |
| sklearnmodels.tree[pandas] | vowel                                  |        0.0909091 |   0.238095    | 0.0042055   |       990 |         12 |
| sklearnmodels.tree[pandas] | isolet                                 |        0.750417  |   2.50749     | 0.055923    |      7797 |         32 |
| sklearnmodels.tree[pandas] | analcatdata_authorship                 |        0.953627  |   0.264641    | 0.00480378  |       841 |         32 |
| sklearnmodels.tree[pandas] | analcatdata_dmft                       |        0.303639  |   0.0138362   | 0.00591637  |       797 |          4 |
| sklearnmodels.tree[pandas] | mnist_784                              |        0.729029  |   1.88539     | 0.489357    |     70000 |         32 |
| sklearnmodels.tree[pandas] | pc4                                    |        0.91358   |   0.23426     | 0.00729949  |      1458 |         32 |
| sklearnmodels.tree[pandas] | pc3                                    |        0.90723   |   0.152112    | 0.00703261  |      1563 |         32 |
| sklearnmodels.tree[pandas] | jm1                                    |        0.806523  |   0.0319509   | 0.0413867   |     10885 |         21 |
| sklearnmodels.tree[pandas] | kc2                                    |        0.871648  |   0.0842077   | 0.00239288  |       522 |         21 |
| sklearnmodels.tree[pandas] | kc1                                    |        0.855382  |   0.0372199   | 0.00838804  |      2109 |         21 |
| sklearnmodels.tree[pandas] | pc1                                    |        0.930568  |   0.0333064   | 0.00441644  |      1109 |         21 |
| sklearnmodels.tree[pandas] | Bioresponse                            |        0.666489  |   0.164744    | 0.0171994   |      3751 |         32 |
| sklearnmodels.tree[pandas] | wdbc                                   |        0.947276  |   0.0895011   | 0.00282231  |       569 |         30 |
| sklearnmodels.tree[pandas] | adult                                  |        0.760718  |   6.51463     | 0.199408    |     48842 |         14 |
| sklearnmodels.tree[pandas] | phoneme                                |        0.754071  |   0.0131485   | 0.0249056   |      5404 |          5 |
| sklearnmodels.tree[pandas] | qsar-biodeg                            |        0.81327   |   0.308156    | 0.00602916  |      1055 |         32 |
| sklearnmodels.tree[pandas] | wall-robot-navigation                  |        0.920088  |   0.244811    | 0.0291749   |      5456 |         24 |
| sklearnmodels.tree[pandas] | semeion                                |        0.767734  |   1.01293     | 0.0106838   |      1593 |         32 |
| sklearnmodels.tree[pandas] | ilpd                                   |        0.713551  |   0.0265674   | 0.0024971   |       583 |         10 |
| sklearnmodels.tree[pandas] | madelon                                |        0.618462  |   0.101102    | 0.0111263   |      2600 |         32 |
| sklearnmodels.tree[pandas] | nomao                                  |        0.714377  | 161.664       | 0.141761    |     34465 |         61 |
| sklearnmodels.tree[pandas] | ozone-level-8hr                        |        0.936859  |   0.130265    | 0.0108487   |      2534 |         32 |
| sklearnmodels.tree[pandas] | cnae-9                                 |        0.817593  |   0.626135    | 0.00771662  |      1080 |         32 |
| sklearnmodels.tree[pandas] | first-order-theorem-proving            |        0.518797  |   1.2263      | 0.0412096   |      6118 |         32 |
| sklearnmodels.tree[pandas] | banknote-authentication                |        0.931487  |   0.0137267   | 0.00671601  |      1372 |          4 |
| sklearnmodels.tree[pandas] | blood-transfusion-service-center       |        0.762032  |   0.00296432  | 0.00279     |       748 |          4 |
| sklearnmodels.tree[pandas] | PhishingWebsites                       |        0.888919  |   0.0934739   | 0.0666966   |     11055 |         30 |
| sklearnmodels.tree[pandas] | cylinder-bands                         |        0.577778  |   0.0332358   | 0.00249133  |       540 |         37 |
| sklearnmodels.tree[pandas] | bank-marketing                         |        0.883015  |   1.97116     | 0.216174    |     45211 |         16 |
| sklearnmodels.tree[pandas] | GesturePhaseSegmentationProcessed      |        0.517877  |   1.01843     | 0.0637478   |      9873 |         32 |
| sklearnmodels.tree[pandas] | har                                    |        0.81105   |   0.97803     | 0.0633139   |     10299 |         32 |
| sklearnmodels.tree[pandas] | dresses-sales                          |        0.592     |   0.0194453   | 0.00306972  |       500 |         12 |
| sklearnmodels.tree[pandas] | texture                                |        0.874545  |   0.923035    | 0.0355575   |      5500 |         32 |
| sklearnmodels.tree[pandas] | connect-4                              |        0.685096  |   0.767395    | 0.447583    |     67557 |         42 |
| sklearnmodels.tree[pandas] | MiceProtein                            |        0.841667  |   0.65681     | 0.00694359  |      1080 |         32 |
| sklearnmodels.tree[pandas] | steel-plates-fault                     |        0.765585  |   0.521218    | 0.0124055   |      1941 |         27 |
| sklearnmodels.tree[pandas] | climate-model-simulation-crashes       |        0.948148  |   0.0634521   | 0.00239673  |       540 |         18 |
| sklearnmodels.tree[pandas] | wilt                                   |        0.946063  |   0.00168219  | 0.0133174   |      4839 |          5 |
| sklearnmodels.tree[pandas] | car                                    |        0.819444  |   0.0104259   | 0.0107318   |      1728 |          6 |
| sklearnmodels.tree[pandas] | segment                                |        0.899134  |   0.155715    | 0.013108    |      2310 |         16 |
| sklearnmodels.tree[pandas] | mfeat-pixel                            |        0.867     |   0.953759    | 0.0128935   |      2000 |         32 |
| sklearnmodels.tree[pandas] | Fashion-MNIST                          |        0.726457  |   1.69787     | 0.467065    |     70000 |         32 |
| sklearnmodels.tree[pandas] | jungle_chess_2pcs_raw_endgame_complete |        0.721145  |   0.0307077   | 0.227125    |     44819 |          6 |
| sklearnmodels.tree[pandas] | numerai28.6                            |        0.50517   |   0.0495775   | 0.264506    |     96320 |         21 |
| sklearnmodels.tree[pandas] | Devnagari-Script                       |        0.366261  |   3.88793     | 0.69042     |     92000 |         32 |
| sklearnmodels.tree[pandas] | CIFAR_10                               |        0.3036    |   1.87418     | 0.400869    |     60000 |         32 |
| sklearnmodels.tree[pandas] | Internet-Advertisements                |        0.914303  |   9.94966     | 0.0331127   |      3279 |       1558 |
| sklearnmodels.tree[pandas] | dna                                    |        0.886064  |   0.868212    | 0.0225196   |      3186 |        180 |
| sklearnmodels.tree[pandas] | churn                                  |        0.8586    |   0.750717    | 0.0206998   |      5000 |         20 |
