import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape(-1,1)
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x, y)
print(model.coef_)
print(model.intercept_)

print(model.predict(x)) # preditores do y

"""
r²: valor entre 0 e 1 em porcentagem, quanto maior, menor o ssr, aqui que sabe se o erro está aceitável
ssr: sempre na mão
"""

rsq = model.score(x,y)
print(rsq)

# print(model.predict(y) - y)

def ssr():
    result = sum(np.square(model.predict(x) - y))
    print(result)

ssr()