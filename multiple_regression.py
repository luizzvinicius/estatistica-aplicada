import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression


from prova1 import teste_normalidade
DEFAULT_VALUE = 0.05

# cars = pd.read_csv("mtcars.csv")
# # cars["disp"],cars["hp"]
# sns.scatterplot(data=cars, x=cars["mpg"], y=cars["cyl"])
# # plt.show()

# sns.scatterplot(data=cars, x=cars["mpg"], y=cars["disp"])
# # plt.show()

# sns.scatterplot(data=cars, x=cars["mpg"], y=cars["hp"])
# # plt.show()

# teste_normalidade(cars["mpg"]) # normal p_value 
# car_p_value = 0.1229
# teste_normalidade(cars["cyl"]) # nâo normal p_value 0.0000
# cyl_p_value = 0
# teste_normalidade(cars["disp"]) # nâo normal p_value 0.0208
# disp_p_value = 0.0208
# teste_normalidade(cars["hp"]) # nâo normal p_value 0.0488
# hp_p_value = 0.0488

def dispersal_index(index_col1, index_col2, col1, col2):
    dispersal_index = 0
    if index_col1 < DEFAULT_VALUE or index_col2 < DEFAULT_VALUE:
        dispersal_index = scipy.stats.spearmanr(col1, col2)
        print("spearmanr")
    else:
        dispersal_index = scipy.stats.pearsonr(col1, col2)
        print("pearsonr")

    return dispersal_index



# print(dispersal_index(car_p_value, cyl_p_value, cars["mpg"], cars["cyl"])) # analisar o statistics
# print(dispersal_index(car_p_value, disp_p_value, cars["mpg"], cars["disp"]))
# print(dispersal_index(car_p_value, hp_p_value, cars["mpg"], cars["hp"]))



# model = LinearRegression()
# model.fit(cars[["cyl","disp","hp"]], cars["mpg"])
# print(model.coef_)
# print(model.intercept_)
# print(model.predict(cars[["cyl","disp","hp"]])) # preditores do y

# # a função fica: y = 34.18 - 1.22x1 - 0.01x2 - 0.019x3

# rsq = model.score(cars[["cyl","disp","hp"]], cars["mpg"])
# print(rsq)

# segunda base
cerveja = pd.read_csv("consumo_cerveja.csv")
# input temperatura (todas) | output consumo de cerveja em L

teste_normalidade(cerveja["Consumo"]) # não é normal
consumo_p_value = 0.0058

teste_normalidade(cerveja["TempMin"]) # não é normal
temp_min_p_value = 0.0005

teste_normalidade(cerveja["TempMedia"]) # não é normal
temp_media_p_value = 0.0341

teste_normalidade(cerveja["TempMax"]) # não é normal
temp_max_p_value = 0.0834

print(dispersal_index(temp_min_p_value, consumo_p_value, cerveja["TempMin"], cerveja["Consumo"]))
print(dispersal_index(temp_media_p_value, consumo_p_value, cerveja["TempMedia"], cerveja["Consumo"]))
print(dispersal_index(temp_max_p_value, consumo_p_value, cerveja["TempMax"], cerveja["Consumo"]))

model = LinearRegression()
model.fit(cerveja[["TempMin","TempMedia","TempMax"]], cerveja["Consumo"])
print(model.coef_)
print(model.intercept_)
print(model.predict(cerveja[["cyl","disp","hp"]])) # preditores do y
