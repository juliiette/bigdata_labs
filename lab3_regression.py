import matplotlib.pyplot as plot
import pandas as pd
from scipy import stats
import seaborn as sns

nyc = pd.read_csv('temp.csv')
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc.Date = nyc.Date.floordiv(100)

pd.set_option('precision', 2)
print(nyc.Temperature.describe())


def predicted_temp(year):
    linear_regression = stats.linregress(x=nyc.Date, y=nyc.Temperature)
    m = linear_regression.slope
    b = linear_regression.intercept
    return m*year + b


print("Predicted:\n2019: ", predicted_temp(2019))
print("2020: ", predicted_temp(2020))
print("2021: ", predicted_temp(2021))
print("1880: ", predicted_temp(1880))

sns.set_style('whitegrid')
axes = sns.regplot(x=nyc.Date, y=nyc.Temperature)

axes.set_ylim(10, 70)
plot.show()

