import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression


from prova1 import teste_normalidade


def dispersal_index(index_col1, index_col2, col1, col2):
    dispersal_index = 0
    if index_col1 < 0.05 or index_col2 < 0.05:
        dispersal_index = scipy.stats.spearmanr(col1, col2)
    else:
        dispersal_index = scipy.stats.pearsonr(col1, col2)

    return dispersal_index


cars = pd.read_csv("mtcars.csv")

car_p_value = teste_normalidade(cars["mpg"])  # normal p_value
cyl_p_value = teste_normalidade(cars["cyl"])  # nâo normal p_value 0.0000
disp_p_value = teste_normalidade(cars["disp"])  # nâo normal p_value 0.0208
hp_p_value = teste_normalidade(cars["hp"])  # nâo normal p_value 0.0488


print(dispersal_index(car_p_value, cyl_p_value, cars["mpg"], cars["cyl"]))  # analisar o statistics
print(dispersal_index(car_p_value, disp_p_value, cars["mpg"], cars["disp"]))
print(dispersal_index(car_p_value, hp_p_value, cars["mpg"], cars["hp"]))


model = LinearRegression()
model.fit(cars[["cyl", "disp", "hp"]], cars["mpg"])
print(model.coef_)
print("intercept", model.intercept_)
# print(model.predict(cars[["cyl","disp","hp"]])) # preditores do y

# a função fica: y = 34.18 - 1.22x1 - 0.01x2 - 0.019x3
# y = intercept + (coef x variável q vc quer)

rsq = model.score(cars[["cyl", "disp", "hp"]], cars["mpg"])
print(rsq)

# segunda base
cerveja = pd.read_csv("consumo_cerveja.csv")
# input temperatura (todas) | output consumo de cerveja em L

consumo_p_value = teste_normalidade(cerveja["Consumo"])  # não é normal
temp_min_p_value = teste_normalidade(cerveja["TempMin"])  # não é normal
temp_media_p_value = teste_normalidade(cerveja["TempMedia"])  # não é normal
temp_max_p_value = teste_normalidade(cerveja["TempMax"])  # é normal

print(dispersal_index(temp_min_p_value, consumo_p_value, cerveja["TempMin"], cerveja["Consumo"]))
print(dispersal_index(temp_media_p_value, consumo_p_value, cerveja["TempMedia"], cerveja["Consumo"]))
print(dispersal_index(temp_max_p_value, consumo_p_value, cerveja["TempMax"], cerveja["Consumo"]))

model = LinearRegression()
model.fit(cerveja[["TempMin", "TempMedia", "TempMax"]], cerveja["Consumo"])
print(model.coef_)
print(model.intercept_)
# print(model.predict(cerveja[["TempMin","TempMedia","TempMax"]])) # preditores do y

rsq = model.score(cerveja[["TempMin", "TempMedia", "TempMax"]], cerveja["Consumo"])
print(rsq)
