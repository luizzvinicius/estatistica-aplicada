import statistics as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

iris = pd.read_csv("Iris.csv")
species = pd.unique(iris["Species"]).tolist()


# Questão 1: maior média de comprimento de pétala entre espécies
def petal_length_mean():
    for specie in species:
        filtered_csv = iris.loc[iris["Species"] == specie]
        petal_length = filtered_csv["PetalLengthCm"]
        print(f"Média de comprimento da pétala na espécie {specie}: {np.array(petal_length).mean() :.2f}cm")


petal_length_mean()

# Questão 2: há uma correlação significativa do tamanho da pétala e da sépala
DEFAULT_VALUE = 0.05
def dispersal():
    for specie in species:
        print(f"Espécie: {specie}")
        filtered_csv = iris.loc[iris["Species"] == specie]
        sepal = filtered_csv["SepalLengthCm"]
        petal = filtered_csv["PetalLengthCm"]

        sepal_pvalue = scipy.stats.shapiro(sepal).pvalue
        petal_pvalue = scipy.stats.shapiro(petal).pvalue
        print(f"Shapiro pvalue sepal: {sepal_pvalue :.2f}")
        print(f"Shapiro pvalue petal: {petal_pvalue :.2f}")

        dispersal_index = 0
        if sepal_pvalue < DEFAULT_VALUE or petal_pvalue < DEFAULT_VALUE:
            dispersal_index = scipy.stats.spearmanr(sepal, petal)
            print("Utilizando spearmanr")
        else:
            dispersal_index = scipy.stats.pearsonr(sepal, petal)
            print("Utilizando pearsonr")

        print(f"Índice de correlação: {dispersal_index.statistic :.2f}")
        print("-" * 50)

        sns.scatterplot(data=iris, x=sepal, y=petal)
        plt.show()


dispersal()

# Questão 3: qual a distribuiçâo das espécies
def show_column_info(database: pd.DataFrame, colum_names: str): # função auxiliar
    column_values = []
    for column in colum_names:
        percentile = np.percentile(database[column], [25, 50, 75])
        mean = np.mean(database[column])
        mode = st.multimode(database[column])
        std = np.std(database[column], ddof=1)

        column_value = database[column].values
        column_values.append(column_value)
        inf, sup = mean - std * 3, mean + std * 3
        outliers = list(database[column][(column_value < inf) | (column_value > sup)])

        print(f"Média {column}: {mean :.2f}")
        print(f"Moda {column}: {mode}")
        print(f"Mediana {column}: {percentile[1]}")
        print(f"Desvio padrão {column}: {std :.2f}")
        print(f"Amplitude interquartil {column}: {percentile[2] - percentile[0] :.2f}")
        print(f"{column} outliers ({len(outliers)}): {outliers}")
        print("-" * 40)

    sns.boxplot(column_values, orient="h")
    plt.yticks([name for name in range(len(colum_names))], colum_names)
    plt.show()


def distribution_of_data():
    columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    infos = {"Iris-setosa": [], "Iris-versicolor": [], "Iris-virginica": []}
    for specie in species:
        print(specie)
        filtered_csv = iris.loc[iris["Species"] == specie]

        for column in columns:
            column_pvalue = scipy.stats.shapiro(filtered_csv[column]).pvalue

            if column_pvalue < DEFAULT_VALUE:
                infos[specie].append("assimétrico")
            else:
                infos[specie].append("simétrico")

        show_column_info(filtered_csv, columns)
    print(infos)

distribution_of_data()

# Questão 4 quais características tem a maior variabilidades
# interpretação dos dados e gráficos gerados acima
