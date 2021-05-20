import matplotlib.pyplot as plot
import pandas as pd

grades = pd.Series([10, 2, 200])

titanic = pd.read_csv('TitanicSurvival.csv')

titanic.columns =['name', 'survived', 'sex', 'age', 'class']
pd.set_option('precision', 2)
#print(titanic.tail())

print("Junior passenger: \n", titanic.loc[titanic[['age']].idxmin()])

print("\nSenior passenger: \n", titanic.loc[titanic[['age']].idxmax()])

print("\nAverage age: \n", titanic[['age']].mean())

women = titanic.loc[(titanic['survived'].str.contains('yes')) & (titanic['class'].str.contains('1st')) & (titanic['sex'].str.contains('female'))]
print('\nJunior women: \n', women.loc[women[['age']].idxmin()])

print('\nSenior women: \n', women.loc[women[['age']].idxmax()])

print("\nSurvived: \n", women.count())
print(titanic.loc[(titanic['survived'].str.contains('yes')) & (titanic['class'].str.contains('1st')) & (titanic['sex'].str.contains('female'))])

titanic.hist(bins=80)
plot.show()
