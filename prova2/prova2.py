import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression

data = pd.read_csv("prova2/BostonHousing.csv")


# Funções auxiliares
def dispersal_index(index_col1, index_col2, col1, col2):
    dispersal_index = 0
    if index_col1 < 0.05 or index_col2 < 0.05:
        dispersal_index = scipy.stats.spearmanr(col1, col2)
    else:
        dispersal_index = scipy.stats.pearsonr(col1, col2)

    return dispersal_index


def teste_normalidade(data):
    _, p_value = scipy.stats.shapiro(data)
    print(f"p_value shapiro: {p_value :.4f}", end=" ")

    msg = ""
    if p_value < 0.05:
        msg = "Rejeitar H0 (hipótese nula). A distribuição da amostra não é normal"
    else:
        msg = "Não rejeitar H0. A distribuição da amostra é normal"
    print(msg)
    return float(f"{p_value :.4f}")


# Colunas usadas
quartos = data["rm"]
media_casas = data["medv"]
criminalidade = data["crim"]
idade = data["age"]
grandes_lotes = data["zn"]
centro_empregos = data["dis"]
razao_aluno_professor = data["ptratio"]
indus = data["indus"]
nox = data["nox"]
chas = data["chas"]
tax = data["tax"]
b = data["b"]
lstat = data["lstat"]
rad = data["rad"]

# Questão 1
p_value_quartos = teste_normalidade(quartos)
p_value_media_casas = teste_normalidade(media_casas)
print(dispersal_index(p_value_quartos, p_value_media_casas, data["rm"], data["medv"])) # 0.6335764254337745


# Questão 2
model = LinearRegression()
model.fit(data[["rm"]], media_casas)
print(model.intercept_)
print(model.coef_)

rsq = model.score(data[["rm"]], media_casas)
print(rsq) #0.4835 baixo

# # y = -34.67 + 9.1X1

# Questão 3
p_value_media_casas = teste_normalidade(media_casas)
p_value_criminalidade = teste_normalidade(criminalidade)
p_value_quartos = teste_normalidade(quartos)
p_value_idade = teste_normalidade(idade)
print(dispersal_index(p_value_media_casas, p_value_quartos, media_casas, quartos))
print(dispersal_index(p_value_media_casas, p_value_idade, media_casas, idade))
print(dispersal_index(p_value_media_casas, p_value_criminalidade, media_casas, criminalidade))

model = LinearRegression()
model.fit(data[["crim", "rm", "age"]], media_casas)
print(model.intercept_)
print(model.coef_)

rsq = model.score(data[["crim", "rm", "age"]], media_casas)
print(rsq) # 0.56

# y = -23.6 - 0.21xX1 + 8xX2 - 0.05xX3
# y2 = -0.21
# y3 = 8
# y4 = -0.05

# Questão 4
p_value_media_casas = teste_normalidade(media_casas)
p_value_grandes_lotes = teste_normalidade(grandes_lotes)
p_value_centro_empregos = teste_normalidade(centro_empregos)
p_value_razao_aluno_professor = teste_normalidade(razao_aluno_professor)
print(dispersal_index(p_value_media_casas, p_value_grandes_lotes, media_casas, grandes_lotes))
print(dispersal_index(p_value_media_casas, p_value_centro_empregos, media_casas, centro_empregos))
print(dispersal_index(p_value_media_casas, p_value_razao_aluno_professor, media_casas, razao_aluno_professor))

model = LinearRegression()
model.fit(data[["zn", "dis", "ptratio"]], media_casas)
print(model.intercept_)
print(model.coef_)

rsq = model.score(data[["zn", "dis", "ptratio"]], media_casas)
print(rsq) # 0.29

# y = -55 - 0.06X1 - 0.17X2 + 1.8X3 (-1)
# y2 = 55 + 0.06X1 + 0.17X2 + 0.17 - 1.8X3 + 1.8

# y2 - y = 1.8 + 0.17
# y2 = 1.97 + y

# Questão 5
colum_names = ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat"]
scores = []

p_value_media_casas = teste_normalidade(media_casas)

for index, c in enumerate(colum_names):
    print(c)
    p_value_colum = teste_normalidade(data[c])
    print(dispersal_index(p_value_media_casas, p_value_colum, media_casas, data[c]))

    model = LinearRegression()
    model.fit(data[[c]], media_casas)

    rsq = model.score(data[[c]], media_casas)
    scores.append({c: [model.intercept_, model.coef_, rsq]})
    print("-" * 20)

for i in scores:
    print(i)

# # rm y = -34.7 + 9.1xX1
# # lstat y = 34.55 - 0.95xX1
