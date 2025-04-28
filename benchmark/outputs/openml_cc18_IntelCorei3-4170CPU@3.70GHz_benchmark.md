# Benchmark table
|   Unnamed: 0 | model              | dataset       |   train_accuracy |   train_time |   test_time |   samples |   features |
|-------------:|:-------------------|:--------------|-----------------:|-------------:|------------:|----------:|-----------:|
|            0 | sklearnmodels.tree | kr-vs-kp      |         0.942428 |   0.0492682  | 0.0161811   |      3196 |         36 |
|            1 | sklearnmodels.tree | letter        |         0.6083   |   0.301529   | 0.0983154   |     20000 |         16 |
|            2 | sklearnmodels.tree | balance-scale |         0.7536   |   0.00817074 | 0.00223227  |       625 |          4 |
|            3 | sklearnmodels.tree | mfeat-factors |         0.889    |   2.96695    | 0.00864477  |      2000 |        216 |
|            4 | sklearnmodels.tree | mfeat-fourier |         0.818    |   1.66791    | 0.00920877  |      2000 |         76 |
|            5 | sklearn.tree       | kr-vs-kp      |         0.977472 |   0.00459733 | 0.00240996  |      3196 |         36 |
|            6 | sklearn.tree       | letter        |         0.81965  |   0.0099169  | 0.00103554  |     20000 |         16 |
|            7 | sklearn.tree       | balance-scale |         0.8      |   0.00053778 | 0.000249282 |       625 |          4 |
|            8 | sklearn.tree       | mfeat-factors |         0.874    |   0.0290913  | 0.00604655  |      2000 |        216 |
|            9 | sklearn.tree       | mfeat-fourier |         0.8055   |   0.015281   | 0.000519478 |      2000 |         76 |
|          nan | sklearnmodels.tree | kr-vs-kp      |         0.522215 |   0.00320538 | 0.0282954   |      3196 |         36 |
|          nan | sklearnmodels.tree | letter        |         0.6053   |   4.03917    | 0.615734    |     20000 |         16 |
## Graphs
 All times are specified in seconds
![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_train_accuracy.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_test_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_samples_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_samples_test_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_samples_speedup_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_samples_speedup_test_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_features_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_features_test_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_features_speedup_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_features_speedup_test_time.png)
