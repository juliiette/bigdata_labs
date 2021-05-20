import statistics
import data
import matplotlib.pyplot as plt
import numpy as np

array = data.rand_array

mean = statistics.mean(array)
median = statistics.median(array)
mode = statistics.mode(array)
dispersion = statistics.pvariance(array)
deviation = statistics.stdev(array)

print("Математичне сподівання: ", mean)
print("Медіана: ", median)
print("Мода: ", mode)
print("Дисперсія: ", dispersion)
print("Відхилення: ", deviation)

x = np.arange(1, 101)
y = array
ax = plt.gca()
ax.bar(x, y, color='crimson')
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
plt.show()
