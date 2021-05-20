import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap


nyc = pd.read_csv('temp.csv')
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc.Date = nyc.Date.floordiv(100)
pd.set_option('precision', 2)

x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11)
print(x_test.shape)

linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)


def prediction_model_test():
    predicted = linear_regression.predict(x_test)
    expected = y_test
    for p, e in zip(predicted[::5], expected[::5]):
        print(f'predicted: {p:.2f}, expected: {e:.2f}')


def temp_predict(year):
    predict = (lambda x: linear_regression.coef_ * x + linear_regression.intercept_)
    return predict(year)


print(f'2019: {temp_predict(2019)}')
print(f'1890: {temp_predict(1890)}')


def diagram():
    axes = sns.scatterplot(data=nyc, x='Date', y='Temperature',
                           hue='Temperature', palette='winter', legend=False)
    axes.set_ylim(10, 70)

    x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
    y = temp_predict(x)

    line = plt.plot(x, y)
    plt.show()


diagram()


np.random.seed(1)
x_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(x_xor[y_xor == 1, 0], x_xor[y_xor == 1, 1],
            c='b', marker='x', label='1')
plt.scatter(x_xor[y_xor == -1, 0], x_xor[y_xor == -1, 1],
            c='r', marker='s', label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')


def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', 'ˆ', 'v')
    colors = ('pink', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y == c1, 0],
                    y=x[y == c1, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=c1,
                    edgecolors='black')

    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]
        plt.scatter(x_test[:, 0],
                    x_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=100,
                    label='Test')


def my_svm():
    svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=10.0)
    svm.fit(x_xor, y_xor)
    plot_decision_regions(x_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.show()


my_svm()
