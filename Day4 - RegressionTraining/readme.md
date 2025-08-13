# Sonuç:

1) İplik Tipi, Kumaş Grubu Hammade Kodu özellikleri eğitimden çıkartıldıktan sonra en iyi performans:

    Random Forest Predictions:
    ----------------------------------------
    MSE: 1157.43
    R² Score: 0.9753
    RMSE: 34.02

    First 10 predictions:
    Index Actual     Predicted    Diff
    ----------------------------------------
    0     19.67      25.26        5.59
    1     19.67      25.86        6.19
    2     27.04      21.84        5.20
    3     104.08     47.15        56.94
    4     760.07     764.10       4.03
    5     2.56       3.84         1.28
    6     2.56       3.84         1.28
    7     22.00      43.32        21.32
    8     6.60       27.44        20.84
    9     4.91       7.55         2.64


2) İplik Tipi, Kumaş Grubu Hammade Kodu özellikleri eğitimden çıkartılmadan önce en iyi performans:

    Decision Tree Predictions:
    ----------------------------------------
    MSE: 753.20
    R² Score: 0.9839
    RMSE: 27.44

    First 10 predictions:
    Index Actual     Predicted    Diff
    ----------------------------------------
    0     19.67      31.00        11.33
    1     19.67      21.24        1.57
    2     27.04      24.60        2.44
    3     104.08     45.32        58.77
    4     760.07     780.53       20.46
    5     2.56       2.42         0.14
    6     2.56       2.42         0.14
    7     22.00      37.83        15.83
    8     6.60       33.63        27.03
    9     4.91       4.93         0.02
