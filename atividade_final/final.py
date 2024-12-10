import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import scipy.stats
import itertools


# fazer summary das variáveis
# p_valor menor que 0.05, variável entra no modelo
# retirar variáveis por ordem decrescentes de p_valor
# *função específica para verificar acurácia
# matriz de confusão
# simular três odds ratio (OR) 3 aleatórias (para mais e para menos) para fazer a função


def teste_normalidade(data):
    _, p_value = scipy.stats.shapiro(data)
    return False if p_value < 0.05 else True  # não consegue apenas retornar a condição


db = pd.read_csv("atividade_final/diabetes2.csv")
colunas = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# normalidades = []
# for c in colunas:
#     normalidades.append({c: teste_normalidade(db[c])})
# print(normalidades)  # todas as colunas não têm uma distribuição normal

def regressao(colunas_param):
    x = db[colunas_param]  # dependentes
    y = db["Outcome"] # independente

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=5000, random_state=0).fit(x_treino, y_treino)

    y_pred = model.predict(x_teste)

    # Avaliar o modelo
    # conf_matrix = confusion_matrix(y_teste, y_pred)
    # classif_report = classification_report(y_teste, y_pred)
    accuracy = accuracy_score(y_teste, y_pred)

    # print(conf_matrix)
    # print(classif_report)
    # print(accuracy)
    if accuracy < 0.70:
        return
    
    testes.append([colunas_param, accuracy])


testes = []
for i, v in enumerate(colunas, start=1):
    combinacoes = itertools.combinations(colunas, i)
    for combinacao in combinacoes:
        regressao([*combinacao])

for e in testes:
    print(e, "\n\n")
