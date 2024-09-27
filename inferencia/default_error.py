import math
import numpy as np

# condutividade térmica do aço, btu/h Fahrenheit
array1 = [41.48, 41.6, 41.72, 41.81, 41.86, 41.95, 42.04, 42.18, 42.26, 42.34]
media = np.mean(array1)
desvio_padrao = np.std(array1, ddof=1)


def default_error(array):
    desvio_padrao = np.std(array, ddof=1)
    raiz_populacao = math.sqrt(len(array))
    return desvio_padrao / raiz_populacao


print("erro:", default_error(array1))
print("média:", np.mean(array1))

intervalo = [media - 3 * default_error(array1), media + 3 * default_error(array1)]

print("intervalo para média populacional:", intervalo)

bootstrap_means = []

for _ in range(400):
    boot_sample = np.random.choice(array1, size=len(array1))
    bootstrap_means.append(boot_sample.mean())

print("desvio das médias:", np.std(bootstrap_means, ddof=1))
print(np.mean(bootstrap_means))
