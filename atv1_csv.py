import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats
import seaborn as sns
import pandas as pd

stock = pd.read_csv("stock_data.csv")
iris = pd.read_csv("Iris.csv")


def show_column_info(database: pd.DataFrame, colum_names: str):
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
        # no_outliers = list(filter(lambda num: num not in outliers, column_values))

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


# iris, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
# stock, ["Open", "Close", "High", "Low"]
# show_column_info(iris, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])


def time_series(database: pd.DataFrame, date_column_name: str):
    """Válido para o stock apenas porque ele que tem uma coluna de tempo"""
    column_date = database[date_column_name]
    database[date_column_name] = pd.to_datetime(column_date)
    sns.lineplot(data=database, x=date_column_name, y="High")
    sns.lineplot(data=database, x=date_column_name, y="Low")
    plt.show()


# time_series(stock, "Date")


def dispersal():
    """Usado apenas com o Iris (flor)"""
    species = pd.unique(iris["Species"]).tolist()
    for specie in species:
        print(f"Species == {specie}")
        filtered_csv = iris.loc[iris["Species"] == specie]
        x = filtered_csv["SepalLengthCm"]
        y = filtered_csv["SepalWidthCm"]

        dispersal_index = scipy.stats.spearmanr(x, y)
        # spearmanr: usado em dados assimétricos. Cria um ranking para isso
        # pearsonr: usado em dados simétricos. Usa a média para isso
        print(f"Índice de dispersão: {dispersal_index}")
        print("-" * 40)

        sns.scatterplot(data=iris, x=x, y=y)
    plt.show()


def dispersal_individual(name):
    filtered_csv = iris.loc[iris["Species"] == name]
    x = filtered_csv["SepalLengthCm"]
    y = filtered_csv["SepalWidthCm"]

    dispersal_index = scipy.stats.spearmanr(x, y)
    print(f"Índice de dispersão: {dispersal_index}")
    print("-" * 40)

    sns.scatterplot(data=iris, x=x, y=y, hue="Species")
    plt.show()


# dispersal_individual('Iris-setosa')

sns.histplot(iris["SepalLengthCm"])
print(scipy.stats.normaltest(iris["SepalLengthCm"]))
plt.show()

# def heatmap():
# sns.heatmap(iris.drop(columns=["Id", "Species"]), annot=True)
