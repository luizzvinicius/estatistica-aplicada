import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression

iris = pd.read_csv("Iris.csv")
species_dispersal = pd.unique(iris["Species"]).tolist()

def dispersal():
    for specie in species_dispersal:
        print(f"Espécie: {specie}")
        filtered_csv = iris.loc[iris["Species"] == specie]
        # Sepal
        sepal_length = filtered_csv["SepalLengthCm"]
        sepal_width = filtered_csv["SepalWidthCm"]
        sepal_length_pvalue = scipy.stats.shapiro(sepal_length).pvalue
        sepal_width_pvalue = scipy.stats.shapiro(sepal_width).pvalue
        # -----------------------
        # Petal
        petal_length = filtered_csv["PetalLengthCm"]
        petal_width = filtered_csv["PetalWidthCm"]
        petal_length_pvalue = scipy.stats.shapiro(petal_length).pvalue
        petal_width_pvalue = scipy.stats.shapiro(petal_width).pvalue
        # -----------------------

        dispersal_index = 0
        dispersal_index_2 = 0
        if sepal_length_pvalue < 0.05 or sepal_width_pvalue < 0.05:
            dispersal_index = scipy.stats.spearmanr(sepal_length, sepal_width)
        else:
            dispersal_index = scipy.stats.pearsonr(sepal_length, sepal_width)
        
        if petal_length_pvalue < 0.05 or petal_width_pvalue < 0.05:
            dispersal_index_2 = scipy.stats.spearmanr(petal_length, petal_width)
        else:
            dispersal_index_2 = scipy.stats.pearsonr(petal_length, petal_width)
        print(f"Índice de correlação sepal: {dispersal_index.statistic :.2f}")
        print(f"Índice de correlação petal: {dispersal_index_2.statistic :.2f}")
        print("-" * 50)

# regression_detailed_results  = {}
# species_groups = iris.groupby('Species')

for specie, group in species_groups:
    print("----------------\n", specie)

    sepal_x, sepal_y = group[['SepalLengthCm']], group['SepalWidthCm']
    petal_x, petal_y = group[['PetalLengthCm']], group['PetalWidthCm']

    sepal_model = LinearRegression().fit(sepal_x, sepal_y)
    petal_model = LinearRegression().fit(petal_x, petal_y)

    print("sepal coef:", sepal_model.coef_)
    print("sepal intercept: ", sepal_model.intercept_)

    print("petal coef:", petal_model.coef_)
    print("petal intercept: ", petal_model.intercept_)
    # Cálculo de R^2 e SSR para sépala
    sepal_r_squared = sepal_model.score(sepal_x, sepal_y)
    sepal_ssr = sum(np.square(sepal_model.predict(sepal_x) - sepal_y))  # Cálculo do SSR

    # Cálculo de R^2 e SSR para pétala
    petal_r_squared = petal_model.score(petal_x, petal_y)
    petal_ssr = sum(np.square(petal_model.predict(petal_x) - petal_y))  # Cálculo do SSR

    # Armazenar resultados detalhados

    print(f"sepal_regression: \nr_squared_percent: {sepal_r_squared}\nssr: {sepal_ssr}")
    print(f"petal_regression: \nr_squared_percent: {petal_r_squared}\nssr: {petal_ssr}")

dispersal()