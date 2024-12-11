import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.stats
import itertools


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

normalidades = []
for c in colunas:
    normalidades.append({c: teste_normalidade(db[c])})
print(normalidades)  # todas as colunas não têm uma distribuição normal


def regressao(colunas_param):
    x = db[colunas_param]  # independentes
    y = db["Outcome"]  # dependente

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=5000, random_state=0).fit(x_treino, y_treino)
    y_pred = model.predict(x_teste)

    # Avaliar o modelo
    conf_matrix = confusion_matrix(y_teste, y_pred)
    accuracy = accuracy_score(y_teste, y_pred)

    if accuracy < 0.7662337662337663:
        return 0

    print(conf_matrix)

    testes.append(
        {
            "colunas": colunas_param,
            "acuracia": accuracy,
            "intercept": model.intercept_,
            "coeficientes": model.coef_,
        }
    )
    return accuracy


testes = []
maior = 0
for i, v in enumerate(colunas, start=1):
    combinacoes = itertools.combinations(colunas, i)
    for combinacao in combinacoes:
        current_accuracy = regressao([*combinacao])
        if current_accuracy > maior:
            maior = current_accuracy

for i, e in enumerate(testes, start=1):
    print(i, e, "\n\n")

print(maior)
