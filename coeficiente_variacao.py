import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

#modelo de trabalho síncrono
data = np.array([94, 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2, 87.9, 87.9, 93.5, 82.3, 79.3, 78.3, 71.6, 88.6, 74.6, 74.1, 80.6])

# print(math.floor(min(data)), round(max(data)))

#modelo de trabalho assíncrono
data_2 = np.array([77.1, 71.7, 91, 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2])

# bins: número pra dividir intervalo
# range: é o range né 

# plt.hist(data, 5, (round(min(data)), 95))
# plt.show()


# plt.hist(data_2, 5, (65, 95))
# plt.show()

# a função utiliza 
# sns.boxplot(data) # 
# plt.show()
# sns.boxplot(data_2) # 
# plt.show()

# sns.boxplot([data, data_2])
# plt.xticks([0,1], ['sync', 'async'])
# plt.xlabel(['work type'])
# plt.ylabel('houry')
# plt.show()

'''
# regra empírica
s = desvio padrão

68% -> [média - s (desvio padrão), média + s ]
95% -> [média - 2s, média + 2s]
99,7% -> [média - 3s, média + 3s ] (ponto de corte)
Se a distribuição for normal, os dados vão estar nesse intervalo

cheryshev (não sei)
75% -> [média - 2s, média + 2s]
88,9% -> [média - 3s, média + 3s]


Não retira todos os outliers de primeira, tem que ser aos poucos para 
'''


stock = pd.read_csv('stock_data.csv')
iris = pd.read_csv('iris.csv')

# sns.boxplot(stock['Open'])
x = np.mean(data)
ponto_corte = np.std(data, ddof=1) * 3
inf, sup = x - ponto_corte, x + ponto_corte
outliers = data[(data < inf) | (data > inf)]
print(outliers)

sns.boxplot(data_2)
plt.show()
