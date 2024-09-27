import math
import numpy as np

# condutividade térmica do aço, btu/h Fahrenheit
array1 = [41.6, 41.48, 42.34, 41.95, 41.86, 42.18, 41.72, 42.26, 41.81, 42.04]

media = np.mean(array1)
desvio_padrao = np.std(array1, ddof=1)

def default_error(array):
    desvio_padrao = np.std(array, ddof=1)
    raiz_populacao = math.sqrt(len(array))
    return desvio_padrao / raiz_populacao

print("erro:", default_error(array1))
print("média:", np.mean(array1))

desvios = [media - 3 * default_error(array1), media + 3 * default_error(array1)]

print("intervalo para média populacional", desvios) # possível intervalo para média da população