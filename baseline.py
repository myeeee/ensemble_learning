import re
import numpy as np
import pandas as pd
import support
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

if __name__ == '__main__':
    # ベンチマーク用のモデル
    models = [
        ('SVM', SVC(random_state=1), SVR()),
        ('GaussianProcess', GaussianProcessClassifier(random_state=1),
         GaussianProcessRegressor(normalize_y=True, alpha=1, random_state=1)),
        ('KNeighbors', KNeighborsClassifier(), KNeighborsRegressor()),
        ('MLP', MLPClassifier(random_state=1, max_iter=1000),
         MLPRegressor(hidden_layer_sizes=5, solver='lbfgs', random_state=1, max_iter=1000)),
    ]

    # 分類・回帰用のデータファイル
    classifier_files = ['iris.data', 'sonar.all-data', 'glass.data']
    classifier_params = [(',', None, None), (',', None, None), (',', None, 0)]
    regressor_files = ['airfoil_self_noise.dat', 'winequality-red.csv', 'winequality-white.csv']
    regressor_params = [(r'\t', None, None), (';', 0, None), (';', 0, None)]

    classifier_files = list(map((lambda x : 'dataset/' + x), classifier_files))
    regressor_files = list(map((lambda x: 'dataset/' + x), regressor_files))

    # 結果表示用のdataframe
    result = pd.DataFrame(columns=['target', 'function'] + [m[0] for m in models],
                          index=range(len(classifier_files + regressor_files) * 2))

    # 分類アルゴリズムの評価
    ncol = 0
    for i, (c, p) in enumerate(zip(classifier_files, classifier_params)):
        # ファイル読み込み
        df = pd.read_csv(c, sep=p[0], header=p[1], index_col=p[2], engine='python')

        # データ生成
        x = df[df.columns[:-1]].values
        y, clz = support.clz_to_prob(df[df.columns[-1]])

        # 結果表示用のtable
        result.loc[ncol, 'target'] = re.split(r'[/._]', c)[1]
        result.loc[ncol + 1, 'target'] = ''
        result.loc[ncol, 'function'] = 'F1Score'
        result.loc[ncol + 1, 'function'] = 'Accuracy'

        for l, c_m, r_m in models:
            kf = KFold(n_splits=5, random_state=1, shuffle=True)
            s = cross_validate(c_m, x, y.argmax(axis=1), cv=kf, scoring=('f1_weighted', 'accuracy'))
            result.loc[ncol, l] = np.mean(s['test_f1_weighted'])
            result.loc[ncol + 1, l] = np.mean(s['test_accuracy'])

        ncol += 2

    # 回帰アルゴリズムの評価
    for i, (c, p) in enumerate(zip(regressor_files, regressor_params)):
        # ファイル読み込み
        df = pd.read_csv(c, sep=p[0], header=p[1], index_col=p[2], engine='python')

        # データ生成
        x = df[df.columns[:-1]].values
        y = df[df.columns[-1]].values.reshape((-1, ))

        # 結果表示用のtable
        result.loc[ncol, 'target'] = re.split(r'[/._]', c)[1]
        result.loc[ncol + 1, 'target'] = ''
        result.loc[ncol, 'function'] = 'R2Score'
        result.loc[ncol + 1, 'function'] = 'MeanSquared'

        for l, c_m, r_m in models:
            kf = KFold(n_splits=5, random_state=1, shuffle=True)
            s = cross_validate(r_m, x, y, cv=kf, scoring=('r2', 'neg_mean_squared_error'))
            result.loc[ncol, l] = np.mean(s['test_r2'])
            result.loc[ncol + 1, l] = -np.mean(s['test_neg_mean_squared_error'])

        ncol += 2

    print(result)
    result.to_csv('baseline.csv', index=False)

