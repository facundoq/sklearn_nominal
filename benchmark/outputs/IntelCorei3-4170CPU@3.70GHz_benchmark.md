# Benchmark table
| model              | dataset       |   train_accuracy |   train_time |   test_time |   samples |   features |
|:-------------------|:--------------|-----------------:|-------------:|------------:|----------:|-----------:|
| sklearn.tree       | kr-vs-kp      |         0.940864 |   0.00438604 | 0.00255197  |      3196 |         36 |
| sklearn.tree       | letter        |         0.60975  |   0.00895747 | 0.00139382  |     20000 |         16 |
| sklearn.tree       | balance-scale |         0.7984   |   0.00064349 | 0.000286467 |       625 |          4 |
| sklearn.tree       | mfeat-factors |         0.8705   |   0.0344792  | 0.00876911  |      2000 |        216 |
| sklearnmodels.tree | kr-vs-kp      |         0.942428 |   0.056276   | 0.0169341   |      3196 |         36 |
| sklearnmodels.tree | letter        |         0.65695  |   0.90728    | 0.0925713   |     20000 |         16 |
| sklearnmodels.tree | balance-scale |         0.7536   |   0.0109974  | 0.00238561  |       625 |          4 |
| sklearnmodels.tree | mfeat-factors |         0.909    |  11.9751     | 0.0107861   |      2000 |        216 |
## Graphs
 All times are specified in seconds
![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_train_accuracy.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_test_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_samples_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_samples_test_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_samples_speedup_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_samples_speedup_test_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_features_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_features_test_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_features_speedup_train_time.png)![alt](openml_cc18_IntelCorei3-4170CPU@3.70GHz_features_speedup_test_time.png)
