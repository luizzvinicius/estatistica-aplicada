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


def bootstrap(array, execucoes):
    bootstrap_means = []
    for _ in range(execucoes):
        boot_sample = np.random.choice(array, size=len(array))
        bootstrap_means.append(boot_sample.mean())
    return bootstrap_means

print("desvio das médias:", np.std(bootstrap(array1, 400), ddof=1))
print(np.mean(bootstrap(array1, 400)))

# tempo para processar pedidos de emprestimo em h
array2 = [24.1514, 27.4145, 20.4, 22.5151, 28.5152, 28.5611, 21.2489, 20.9983, 24.984, 22.6245]
media2 = np.mean(array2)
erro_padrao2 = default_error(array2)
print(erro_padrao2)

print("intervalo para média", [media2 - 3 * erro_padrao2, media2 + 3 * erro_padrao2])

print(np.mean(bootstrap(array2, 400)))
