import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt

iris = pd.read_csv("Iris.csv")

# questâo 1: maior média de comprimento de pétala entre espécies
medias = []
species = pd.unique(iris["Species"]).tolist()

for specie in species:
    filtered_csv = iris.loc[iris["Species"] == specie]
    petal_length = filtered_csv["PetalLengthCm"].tolist()
    medias.append({specie: np.array(petal_length).mean()})

# mostrar centimetros
# print(medias)

# questâo 2 há uma correlação significativa do tamanho da pétala e da sépala

DEFAULT_VALUE = 0.05

def dispersal():
    species = pd.unique(iris["Species"]).tolist()
    for specie in species:
        print(f"Species == {specie}")
        filtered_csv = iris.loc[iris["Species"] == specie]
        sepal = filtered_csv["SepalLengthCm"]
        petal = filtered_csv["PetalLengthCm"]

        shapiros = [scipy.stats.shapiro(sepal).pvalue, scipy.stats.shapiro(petal).pvalue]
        print(f"Shapiro sepal: {shapiros[0]}")
        print(f"Shapiro petal: {shapiros[1]}")

        dispersal_index = 0
        if shapiros[0] < DEFAULT_VALUE or shapiros[1] < DEFAULT_VALUE:
            dispersal_index = scipy.stats.spearmanr(sepal, petal) # avaliar statistics
        else:
            dispersal_index = scipy.stats.pearsonr(sepal, petal)

        print(f"Índice de dispersão: {dispersal_index.statistic}") # próximo de 1 muito relacionados, próximo de 0 não há influência

        # spearmanr: usado em dados assimétricos. Cria um ranking para isso
        # pearsonr: usado em dados simétricos. Usa a média para isso
        print("-" * 40)
        sns.scatterplot(data=iris, x=sepal, y=petal)
        plt.show()

# dispersal()

# Questâo 3 qual a distribuiçâo das espécies




# Questâo 4 quais características tem a maior variabilidades

