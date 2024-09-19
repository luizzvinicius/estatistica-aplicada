import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

stock = pd.read_csv("stock_data.csv")
iris = pd.read_csv("Iris.csv")


def show_column_info(database: pd.DataFrame, colum_name: str):
    percentile = np.percentile(database[colum_name], [25, 50, 75])
    mode = st.multimode(database[colum_name])

    initial = database[colum_name].values
    values_to_show = [initial]
    outliers = []
    while True:
        mean = np.mean(initial)
        std = np.std(initial, ddof=1)
        inf, sup = mean - std * 3, mean + std * 3

        outlier = initial[(initial < inf) | (initial > sup)]
        outliers.append(outlier)
        no_outliers = list(filter(lambda num: num not in outlier, initial))
        values_to_show.append(no_outliers)
        initial = np.array(no_outliers)

        if len(outlier) > 0:
            continue
        break
    values_to_show.pop() # remove o último porque é repetido

    print(f"Média {colum_name}: {mean :.2f}")
    print(f"Moda {colum_name}: {mode}")
    print(f"Mediana {colum_name}: {percentile[1]}")
    print(f"Desvio padrão {colum_name}: {std :.2f}")
    print(f"Amplitude interquartil {colum_name}: {percentile[2] - percentile[0]}")
    print(f"{colum_name} outliers ({len(outliers)}): {outliers}")
    # print(min(column_values), max(column_values))
    # plt.hist(column_values, 5, (5, 80))
    sns.boxplot(values_to_show, orient="h", fill=False)
    plt.yticks([0, 1], ["original", "no outliers"])
    plt.show()


show_column_info(stock, "Open")
