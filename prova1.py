"""
Questão pede:
    O tipo de teste que vai ser utilizado, o motivo da escolha e o resultado do teste concluindo o que diz respeito à média das populações

Teste de hipótese da média:
1- Verificar se a distribuição é normal ou não (pValor) shapiro wilk
2- Teste de homogeneidade (levene, verifica a variância) (pValor)

Conforme os resultados de 1 e 2, decidir qual teste utilizar:
Anova: testa se há diferenças significativas entre as médias de três ou mais grupos
    Distribuição normal
    Variâncias homogêneas entre os grupos

T_teste: compara as médias de dois grupos para determinar se elas são significativamente diferentes entre si
    Amostras não pareadas: por exemplo, resultado de indivíduos distintos
    Distribuição normal
    Mesma variância

Mann-Whitney: teste não paramétrico que compara dois grupos para ver se as suas distribuições diferem significativamente
    Duas amostras não pareadas
    Mesma variância
    Normalidade não influencia no teste
    Uso: Quando há duas amostras que não atendem às suposições de normalidade exigidas pelo t-test
    Suposição: Não exige normalidade, funciona bem para dados de escalas ordinais ou amostras pequenas

Kruskal-Wallis: é um teste não paramétrico usado para comparar três ou mais grupos, similar ao ANOVA, mas sem a exigência de normalidade
    Não exige normalidade ou variância homogênea


Wilcoxon Signed Rank (não falou esse no áudio)
    Uso: Quando há duas amostras dependentes/pareadas e os dados não seguem uma distribuição normal


Exemplo de resposta:
    Vamos utilizar o teste ___ porque a distribuição é ___, as variâncias são ___. O pvalue de cada um ... As médias da população serão iguais ou não
"""

from scipy import stats
import scikit_posthocs as sp
import numpy as np


def teste_normalidade(data):
    _, p_value = stats.shapiro(data)
    print(f"p_value shapiro: {p_value :.4f}", end=" ")

    msg = ""
    if p_value < 0.05:
        msg = "Rejeitar H0 (hipótese nula). A distribuição da amostra não é normal"
    else:
        msg = "Não rejeitar H0. A distribuição da amostra é normal"
    print(msg)
    return float(f"{p_value :.4f}")


def levene_test(*data):
    _, p_value = stats.levene(*data)
    print(f"p_value levene: {p_value :.4f}")

    msg = ""
    if p_value < 0.05:
        msg = "Rejeitar H0 (hipótese nula). A variância das amostras são diferentes"
    else:
        msg = "Não rejeitar H0. A variância das amostras são as mesmas"
    print(msg)


def t_teste_ind(*data, equal_var=True):
    _, p_value = stats.ttest_ind(*data, equal_var)
    print(f"p_value t_teste independente: {p_value :.8f}")
    one_sided = p_value / 2
    print(f"Amostra unilateral. p_value unilateral: { one_sided :.4f}")

    msg = ""
    if one_sided < 0.05:
        msg = "As médias são diferentes"
    else:
        msg = "As médias são iguais"
    print(msg)


def t_teste_rel(*data):
    _, p_value = stats.ttest_rel(*data)
    print(f"p_value t_teste relacionado: {p_value :.6f}")
    one_sided = p_value / 2
    print(f"Amostra unilateral. p_value unilateral: { one_sided :.4f}")

    msg = ""
    if one_sided < 0.05:
        msg = "As médias são diferentes"
    else:
        msg = "As médias são iguais"
    print(msg)


def posthoc_ttest(
    *dados, nome_dados=[""]
):  # May be used after a parametric ANOVA to do pairwise comparisons.
    posthoc_df = sp.posthoc_ttest([*dados], equal_var=True, p_adjust="bonferroni")
    posthoc_df.columns = nome_dados
    posthoc_df.index = nome_dados
    print(posthoc_df)


# Interpretar dados do posthoc
# valores pequenos demais: médias das populações são diferentes, e quanto menor o valor, maior a diferença entre as médias
# valores "grandes": médias das populações são iguais ou a diferença entre elas é pouca


def anova(*data):
    _, p_value = stats.f_oneway(*data)
    print(f"p_value anova: {p_value :.6f}")

    msg = ""
    if p_value < 0.05:
        msg = "Pelo menos uma das médias são diferentes"
    else:
        msg = "As médias são iguais"
    print(msg)


def posthoc_man_whitney(*dados, nome_dados=[""]):
    posthoc_df = sp.posthoc_mannwhitney([*dados], p_adjust="bonferroni")
    posthoc_df.columns = nome_dados
    posthoc_df.index = nome_dados
    print(posthoc_df)


def mann_whitney(dados1, dados2):
    _, pvalue = stats.mannwhitneyu(
        dados1, dados2, alternative="two-sided"
    )  # passar array separadamente?
    print(f"p_value mann_whitney: {pvalue :.4f}")

    msg = ""
    if pvalue < 0.05:
        msg = "As médias são diferentes"
    else:
        msg = "As médias são iguais"
    print(msg)


def kruskal_wallis(*data):
    _, p_value = stats.kruskal(*data)
    print(f"p_value kruskal: {p_value :.6f}")

    msg = ""
    if p_value < 0.05:
        msg = "Pelo menos uma das médias são diferentes"
    else:
        msg = "As médias são iguais"
    print(msg)


def wilcoxon(dados1, dados2):  # pvalue é o importante
    _, pvalue = stats.wilcoxon(dados1, dados2)  # alternative default two sided
    print(f"p_value: {pvalue :.6f} | one_tailed_pval: {pvalue / 2 :.6f}")

    # _, one_sided_pvalue = stats.wilcoxon(dados1, dados2, alternative="less")
    # print(f"one sided p_value: {one_sided_pvalue :.6f}")

    msg = ""
    if pvalue < 0.05:
        msg = "As médias são diferentes"
    else:
        msg = "As médias são iguais"
    print(msg)


# Questão 1

grupo_a = np.array([80, 85, 90, 78, 82, 88, 84])
grupo_b = np.array([78, 80, 85, 88, 82, 86, 90])

# teste_normalidade(grupo_a) # a distribuição é normal | pvalue 0.9560
# teste_normalidade(grupo_b) # a distribuição é normal | pvalue 0.9026
# levene 0.9132 | A variância das amostras são as mesmas
# levene_test(grupo_a, grupo_b)

# t_teste_ind(grupo_a, grupo_b) # p_value unilateral: 0.4515 as médias são iguais

"""Após aplicar o teste de Shapiro-Wilk e obter 0.9560 como pvalor do grupo A e 0.9026 para o valor p do grupo B, conclui-se que a distribuição 
de ambas as amostras são normais. Em seguida, aplicou-se o teste de levene, obtendo 0.9132 pra o p valor, concluindo também que a variância das amostras são as mesmas.
Depois desses testes e levando em conta que as amostras não são pareadas, escolhi o tteste, obtendo 0.4515 (unilateral), concluindo que a média de eficácia dos métodos de estudo 
foi a mesma para os grupos A e B."""

# Questão 2 
especie_x = np.array([150, 155, 160, 158, 152, 162])
especie_y = np.array([145, 148, 150, 149, 151, 147])

# teste_normalidade(especie_x) # a distribuição é normal | pvalue 0.8263
# teste_normalidade(especie_y) # a distribuição é normal | pvalue 0.9637
# p_value levene: 0.0493 | A variância das amostras são diferentes
# levene_test(especie_x, especie_y)
# t_teste_ind(especie_x, especie_y, equal_var=False)

"""Após aplicar o teste de Shapiro-Wilk e obter 0.8263 como pvalor da espécie x e 0.9637 para o valor p da espécie y, conclui-se que a distribuição 
de ambas as amostras são normais. Em seguida, aplicou-se o teste de levene, obtendo 0.0493 para o p valor, concluindo também que a variância das amostras são diferentes.
Depois desses testes e levando em conta que as amostras não são pareadas, escolhi o tteste (adaptado com o parametro equal_var = False), obtendo 0.0019 (unilateral), 
concluindo que a média de peso das aves foi diferentes"""

# Questão 3
dieta_a = np.array([5, 7, 6, 8, 5])
dieta_b = np.array([4, 6, 5, 7, 6])
dieta_c = np.array([6, 8, 7, 9, 7])

# teste_normalidade(dieta_a) # a distribuição é 0.4211 | pvalue 
# teste_normalidade(dieta_b) # a distribuição é 0.8140 | pvalue 
# teste_normalidade(dieta_c) # a distribuição é 0.8140 | pvalue 
# p_value levene: 0.9009 | A variância das amostras são as mesmas
# levene_test(dieta_a, dieta_b, dieta_c)

# anova(dieta_a, dieta_b, dieta_c)

"""Após aplicar o teste de Shapiro-Wilk e obter 0.4211 como pvalor da dieta A e 0.8140 para o pvalor das dietas B e C, conclui-se que a distribuição 
de todas as amostras são normais. Em seguida, aplicou-se o teste de levene, obtendo 0.9009 para o p valor, concluindo também que a variância das amostras são as mesmas.
Depois desses testes e levando em conta que as amostras não são pareadas, escolhi o Anova, obtendo 0.091988, concluindo que a média de perda de peso sâo as mesmas"""