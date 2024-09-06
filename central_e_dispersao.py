import numpy as np
import statistics as st

# coeficiente de variação: S (desvio padrão) / x barra

data = np.array([
    94, 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2, 87.9, 87.9, 93.5, 82.3, 79.3, 78.3, 71.6, 88.6, 74.6, 74.1, 80.6
])

data_2 = np.array(
    [78.2, 77.1, 71.7, 91, 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2]
)

media = data.mean()
media_2 = data_2.mean()
print("Média: ", media)
print(f'mediana 1: {np.median(data)}')

print("Média 2: ", media_2)
print(f'mediana 2: {np.median(data_2)}')

percentil1 = np.percentile(data, [ 25, 50, 75 ])
percentil2 = np.percentile(data_2, [ 25, 50, 75 ])
print("Quartil 1 e 3 do data", percentil1)
print("Quartil 1 e 3 do data_2", percentil2)

print(st.mode(data))
print(st.multimode(data_2))

print('amplitude: ', np.ptp(data))
print('amplitude2: ', np.ptp(data_2))

print('variancia data: ', data.var(ddof=1)) # amostra
print('variancia data_2: ', data.var(ddof=0))

print('variancia2 data_2: ', data_2.var(ddof=1)) # população
print('variancia2 data_2: ', data_2.var(ddof=0))

print('desvio padrão', data.std(ddof=1))
