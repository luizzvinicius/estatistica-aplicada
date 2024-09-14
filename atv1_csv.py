import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats
import seaborn as sns
import pandas as pd

stock = pd.read_csv("stock_data.csv")
iris = pd.read_csv("Iris.csv")


def show_column_info(database: pd.DataFrame, colum_name: str):
    column = database[colum_name]
    percentile = np.percentile(database[colum_name], [25, 50, 75])
    mean = np.mean(database[colum_name])
    mode = st.multimode(database[colum_name])
    std = np.std(database[colum_name], ddof=1)

    column_values = database[colum_name].values
    inf, sup = mean - std * 3, mean + std * 3
    outliers = list(column[(column_values < inf) | (column_values > sup)])
    no_outliers = list(filter(lambda num: num not in outliers, column_values))

    print(f"Média {colum_name}: {mean :.2f}")
    print(f"Moda {colum_name}: {mode}")
    print(f"Mediana {colum_name}: {percentile[1]}")
    print(f"Desvio padrão {colum_name}: {std :.2f}")
    print(f"Amplitude interquartil {colum_name}: {percentile[2] - percentile[0]}")
    print(f"{colum_name} outliers ({len(outliers)}): {outliers}")
    print(min(column_values), max(column_values))
    # plt.hist(column_values, 5, (5, 80))
    # sns.boxplot([column_values, no_outliers], orient="h")
    # plt.yticks([0,1], ['original', 'no outliers'])
    
    # Time series
    # stock_copy = stock.copy()
    # stock_copy['Date'] = pd.to_datetime(stock_copy['Date'])
    # sns.lineplot(data=stock_copy, x='Date', y='High')
    # sns.lineplot(data=stock_copy, x='Date', y='Low')


    # Dispersão
    # x = iris['SepalLengthCm']
    # y = iris['SepalWidthCm']
    
    # print(scipy.stats.spearmanr(x, y)) # cria um ranking assimétrico
    # # scipy.stats.pearsonr(x, y) # usa a média métrico
    # # sns.boxplot(x)
    # sns.scatterplot(x=x, y=y)
    # column = iris['Species']
    # print(iris.loc[column == 'Iris-setosa'])
    # plt.show()

    # heatmap
    sns.heatmap(iris, annot=True)
    plt.show()



show_column_info(stock, "Close")
