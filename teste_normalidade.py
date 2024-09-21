import numpy as np
import statistics as st
import scipy.stats

sync = np.array([
    94, 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2, 87.9, 87.9, 93.5, 82.3, 79.3, 78.3, 71.6, 88.6, 74.6, 74.1, 80.6
])

async1 = np.array([77.1, 71.7, 91, 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2])

DEFAULT_VALUE = 0.05

normalidade_sync = scipy.stats.shapiro(sync)
normalidade_async = scipy.stats.shapiro(async1)

normalidades = [normalidade_sync, normalidade_async]

for number in normalidades:
    print(f"pvalue: {number.pvalue}")
    print(f"skew: {scipy.stats.skew(number)}")
    if number.pvalue < DEFAULT_VALUE:
        print(f"Rejeitar H0 em {number}")
    elif number.pvalue == DEFAULT_VALUE:
        print("Jogue na mega-sena")
    else:
        print(f"Nâo rejeitar H0 em {number}")
    # scipy.stats.kurtosis()
