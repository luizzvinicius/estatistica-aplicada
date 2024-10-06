# EXEMPLO 1 (ESTUDO SÍNCRONO E ASSÍNCRONO)

import statistics
import numpy as np
import scikit_posthocs as sp
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
# import scipy as scp

sync = np.array([ 
    94.0, 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2, 87.9, 87.9, 93.5, 82.3, 79.3, 78.3, 71.6, 88.6, 74.6, 74.1, 80.6
])
asyncr = np.array([
    77.1, 71.7, 91.0, 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2
])


print(f"desvio padrão sync: {sync.std()}")
# sync.ptp()
print(f"Média do sync: {sync.mean()}")
print(f"Média do async: {asyncr.mean()}")
print(f"Mediana sync: {statistics.median(sync)}")

"""
TESTE DE NORMALIDADE!
Teste de hipótese de Shapiro-Wilk

H0:Os dados têm distribuição normal
H1:Os dados não têm distribuição normal.

Conclusões:
Se p-valor (valor de prova) <= 0.05, rejeita H0 (H0 é falso) e assume H1 (H1 é verdadeiro);
Se p-valor (valor de prova) > 0.05 não deve rejeitar H0 (H0 é verdadeiro).

0.05 = 5% é a cauda do intervalo de confiança de 95%.
"""


def check_normality(data):
    test_stat_normality, p_value_normality = stats.shapiro(data)
    print(f"p value: {p_value_normality :.4f}")

    if p_value_normality < 0.05:
        print("Reject null hypothesis >> The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis >> The data is normally distributed")


check_normality(sync)
check_normality(asyncr)
sns.distplot(sync)
plt.show()

"""
TESTANDO A VARIABILIDADE DOS DADOS

Aplicando o teste de homogeneidade de Levene

H0:As varâncias são iguais
H1:As variâncias são diferentes.

Esse teste compara a variância das amostras com a variância geral e é usado como um passo inicial antes de realizar testes paramétricos, como o teste t de Student

Conclusões:
Se pvalue <= 0.05, rejeita H0 e assume H1
Se pvalue > 0.05 não deve rejeitar H0.
"""

def levene_test(*data):
    test_stat_var, p_value_var = stats.levene(*data)
    print(f"p value: {p_value_var :.4f}")
    if p_value_var < 0.05:
        print("Reject null hypothesis >> The variances of the samples are different.")
    else:
        print("Fail to reject null hypothesis >> The variances of the samples are same.")

levene_test(sync, asyncr)

"""
PARÂMETROS:
PRÉ-REQUISITOS
DUAS AMOSTRAS NÃO PAREADAS
SÃO NORMALMENTE DISTRIBUIDAS
MESMA VARIÂNCIA

O Teste t compara as médias de dois grupos para determinar se elas são significativamente diferentes entre si.

Uso: Quando você tem duas amostras independentes (ou uma amostra comparada com uma média conhecida) e deseja testar se suas médias são diferentes.
Suposição: As amostras devem seguir uma distribuição normal e ter variâncias semelhantes (dependendo da variação do teste t).

Hipóteses
H0: as médias são iguais
H1: as médias são diferentes

Conclusões:
Se pvalue <= 0.05, rejeita H0 e assume H1;
Se pvalue > 0.05 não deve rejeitar H0.
"""

ttest, p_value = stats.ttest_ind(sync, asyncr)
print(f"p value: {p_value :.8f}")
print(
    f"since the hypothesis is one sided >> use p_value/2 >> p_value_one_sided: {p_value / 2 :.4f}"
)
if p_value / 2 < 0.05:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")


# Conclusão: As médias das notas da população dos alunos que estudaram de forma síncrona e assíncrona são diferentes e
# a média dos alunos com estudo síncrono é maior que a dos alunos com estudo assíncrono.


# EXEMPLO 2 (ALIMENTAÇÃO COM LEITE MATERNO, FÓRMULA OU OS DOIS)

only_breast = np.array(
    [
        794.1,
        716.9,
        993.0,
        724.7,
        760.9,
        908.2,
        659.3,
        690.8,
        768.7,
        717.3,
        630.7,
        729.5,
        714.1,
        810.3,
        583.5,
        679.9,
        865.1,
    ]
)

only_formula = np.array(
    [
        898.8,
        881.2,
        940.2,
        966.2,
        957.5,
        1061.7,
        1046.2,
        980.4,
        895.6,
        919.7,
        1074.1,
        952.5,
        796.3,
        859.6,
        871.1,
        1047.5,
        919.1,
        1160.5,
        996.9,
    ]
)

both = np.array(
    [
        976.4,
        656.4,
        861.2,
        706.8,
        718.5,
        717.1,
        759.8,
        894.6,
        867.6,
        805.6,
        765.4,
        800.3,
        789.9,
        875.3,
        740.0,
        799.4,
        790.3,
        795.2,
        823.6,
        818.7,
        926.8,
        791.7,
        948.3,
    ]
)


check_normality(only_breast)
check_normality(only_formula)
check_normality(both)


sns.distplot(only_formula)
plt.show()

levene_test(only_breast, only_formula, both)


# PARÂMETROS:
# PRÉ-REQUISITOS
# MAIS DE DUAS AMOSTRAS NÃO PAREADAS
# SÃO NORMALMENTE DISTRIBUIDAS
# MESMA VARIÂNCIA

# APLICANDO ANOVA - ANÁLISE DE VARIÂNCIA (TESTE PARAMÉTRICO)

# testa se há diferenças significativas entre as médias de três ou mais grupos.

# Uso: Quando você tem três ou mais grupos independentes e quer testar se ao menos uma das médias é diferente.
# Suposição: As amostras devem seguir uma distribuição normal e ter variâncias homogêneas entre os grupos.

# H0: as médias são iguais;
# H1: pelo menos uma das médias é diferente.

# Conclusões:
# Se pvalue <= 0.05, rejeita H0 e assume H1;
# Se pvalue > 0.05 não deve rejeitar H0.

F, p_value = stats.f_oneway(only_breast, only_formula, both)
print("p value:%.6f" % p_value)
if p_value < 0.05:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")


# Conclusão: Pelo menos uma das médias da população é diferente.


# PARÂMETROS:
# DUAS AMOSTRAS NÃO PAREADAS
# SÃO NORMALMENTE DISTRIBUIDAS
# MESMA VARIÂNCIA

# APLICANDO T-TEST


# O Posthoc t-test é utilizado após um ANOVA quando se detecta uma diferença significativa, para identificar quais grupos são diferentes entre si.

# Uso: Quando o ANOVA detecta diferenças significativas entre grupos, o posthoc t-test compara cada par de grupos individualmente.
# Suposição: Mesmo que o ANOVA tenha identificado uma diferença, o posthoc t-test também exige variância homogênea e dados normalmente distribuídos.

# H0: as médias são iguais
# H1: as médias são diferentes

# Conclusões:
# Se pvalue <= 0.05, rejeita H0 e assume H1;
# Se pvalue > 0.05 não deve rejeitar H0.


posthoc_df = sp.posthoc_ttest(
    [only_breast, only_formula, both], equal_var=True, p_adjust="bonferroni"
)

group_names = ["only breast", "only formula", "both"]
posthoc_df.columns = group_names
posthoc_df.index = group_names
posthoc_df.style.applymap(
    lambda x: "background-color:violet" if x < 0.05 else "background-color: white"
)


print(only_breast.mean())
print(only_formula.mean())
print(both.mean())


# only_formula e only_breast p-valor = 0, as médias das populações são diferentes;
# only_formula e both p-valor = 0, as médias das populações são diferentes;
# only_breast e both p-valor = 0.129, as médias das populações são iguais;

# Conclusão 1: alimentar os bebês com leite materno ou intercalando leite materno e fórmula tem o mesmo aumento médio de peso.
# Conclusão 2: o aumento médio de peso dos bebês que são alimentados com leite artificial (fórmula) será maior que os bebês
# alimentados com leite materno ou com os dois intercalados.


# # EXEMPLO 3 (HORAS EXTRAS POR EQUIPES DE DESENVOLVIMENTO DE SOFTWARE E EQUIPE DE TESTE)

test_team = np.array(
    [6.2, 7.1, 1.5, 2, 3, 2, 1.5, 6.1, 2.4, 2.3, 12.4, 1.8, 5.3, 3.1, 9.4, 2.3, 4.1]
)
developer_team = np.array(
    [
        2.3,
        2.1,
        1.4,
        2.0,
        8.7,
        2.2,
        3.1,
        4.2,
        3.6,
        2.5,
        3.1,
        6.2,
        12.1,
        3.9,
        2.2,
        1.2,
        3.4,
    ]
)


check_normality(test_team)
check_normality(developer_team)


sns.distplot(developer_team)
plt.show()

levene_test(test_team, developer_team)


# PARÂMETROS:
# PRÉ-REQUISITOS
# DUAS AMOSTRAS NÃO PAREADAS
# MESMA VARIÂNCIA

# CONSTATAÇÃO
# NÃO SÃO NORMALMENTE DISTRIBUIDOS (NORMALIDADE NÃO INFLUENCIA NO TESTE)


#  APLICANDO MANN-WHITNEY U TEST (NÃO PARAMÉTRICO)

# O Mann-Whitney U é um teste não paramétrico que compara dois grupos para ver se as suas distribuições diferem significativamente.
# Uso: Quando você tem duas amostras independentes que não atendem às suposições de normalidade exigidas pelo t-test.
# Suposição: Não exige normalidade; funciona bem para dados de escalas ordinais ou amostras pequenas.

# Hipóteses
# H0: as médias são iguais
# H1: as médias são diferentes

# Conclusões:
# Se pvalue <= 0.05, rejeita H0 e assume H1;
# Se pvalue > 0.05 não deve rejeitar H0.

ttest, pvalue = stats.mannwhitneyu(test_team, developer_team, alternative="two-sided")
print("p-value:%.4f" % pvalue)
if pvalue < 0.05:
    print("Reject null hypothesis")
else:
    print("Fail to recejt null hypothesis")


print(developer_team.mean())
print(test_team.mean())


# Conclusão: As médias das horas extras trabalhadas por semana da população de teste é igual a média da
# população de desenvolvimento.


# # EXEMPLO 4 (CONSUMIDORES ATRAIDOS PELAS PLATAFORMAS YOUTUBE, INSTAGRAM E FACEBOOK POR DIA)

youtube = np.array(
    [
        1913,
        1879,
        1939,
        2146,
        2040,
        2127,
        2122,
        2156,
        2036,
        1974,
        1956,
        2146,
        2151,
        1943,
        2125,
    ]
)

instagram = np.array(
    [
        2305.0,
        2355.0,
        2203.0,
        2231.0,
        2185.0,
        2420.0,
        2386.0,
        2410.0,
        2340.0,
        2349.0,
        2241.0,
        2396.0,
        2244.0,
        2267.0,
        2281.0,
    ]
)

facebook = np.array(
    [
        2133.0,
        2522.0,
        2124.0,
        2551.0,
        2293.0,
        2367.0,
        2460.0,
        2311.0,
        2178.0,
        2113.0,
        2048.0,
        2443.0,
        2265.0,
        2095.0,
        2528.0,
    ]
)


check_normality(youtube)
check_normality(instagram)
check_normality(facebook)

levene_test(youtube, instagram, facebook)


# PARÂMETROS:
# PRÉ-REQUISITOS
# MAIS DE DUAS AMOSTRAS NÃO PAREADAS


# CONSTATAÇÃO
# NÃO SÃO NORMALMENTE DISTRIBUIDOS (NORMALIDADE NÃO INFLUENCIA NO TESTE)
# VARIÂNCIAS DIFERENTES (VARIABILIDADE NÃO INFLUENCIA NO TESTE)

# APLICANDO KRUSKAL-WALLIS ANOVA (NÃO PARAMÉTRICO)

# O Kruskal-Wallis é um teste não paramétrico usado para comparar três ou mais grupos, similar ao ANOVA, mas sem a exigência de normalidade.
# Uso: Quando você tem três ou mais grupos independentes e os dados não seguem uma distribuição normal.
# Suposição: Não exige normalidade ou variância homogênea.

# H0: as médias são iguais
# H1: pelo menos uma das médias é diferente

# Conclusões:
# Se pvalue <= 0.05, rejeita H0 e assume H1;
# Se pvalue > 0.05 não deve rejeitar H0.

F, p_value = stats.kruskal(youtube, instagram, facebook)
print("p value:%.6f" % p_value)
if p_value < 0.05:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")


# Conclusão: Existe significância suficiente para afirmar que pelo menos uma das médias da população é diferente.


# APLICANDO MANN-WHITNEY U TEST

# H0: as médias são iguais
# H1: as médias são diferentes

# Conclusões:
# Se pvalue <= 0.05, rejeita H0 e assume H1;
# Se pvalue > 0.05 não deve rejeitar H0.

posthoc_df = sp.posthoc_mannwhitney(
    [youtube, instagram, facebook], p_adjust="bonferroni"
)
group_names = ["youtube", "instagram", "facebook"]
posthoc_df.columns = group_names
posthoc_df.index = group_names
posthoc_df.style.applymap(
    lambda x: "background-color:violet" if x < 0.05 else "background-color: white"
)


print(youtube.mean())
print(instagram.mean())
print(facebook.mean())


# Nível de confiança 95%
# Conclusão 1: Podemos afirmar que a população do facebook e Instagram tem a mesma média de consumidores atraidos diariamente.
# Conclusão 2: Podemos afirmar que a população do Youtube tem média de consumidores atraidos diariamente, menor que a população
# do facebook e do instagram.


# # EXEMPLO 5 (NÍVEL DE COLESTEROL ANTES DA DIETA E DEPOIS DA DIETA)

before_diet = np.array(
    [
        224,
        235,
        223,
        253,
        253,
        224,
        244,
        225,
        259,
        220,
        242,
        240,
        239,
        229,
        276,
        254,
        237,
        227,
    ]
)
after_diet = np.array(
    [
        198,
        195,
        213,
        190,
        246,
        206,
        225,
        199,
        214,
        210,
        188,
        205,
        200,
        220,
        190,
        199,
        191,
        218,
    ]
)


check_normality(before_diet)
check_normality(after_diet)


# T-TEST PARA AMOSTRAS PAREADAS (PARAMÉTRICO)

# PRÉ-REQUISITO
# DISTRIBUIÇÃO NORMAL

# H0: as médias são iguais
# H1: as médias são diferentes

# Conclusões:
# Se pvalue <= 0.05, rejeita H0 e assume H1;
# Se pvalue > 0.05 não deve rejeitar H0.

test_stat, p_value_paired = stats.ttest_rel(before_diet, after_diet)
print("p value:%.6f" % p_value_paired, "one tailed p value:%.6f" % (p_value_paired / 2))
if p_value_paired < 0.05:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")


# Conclusão: Existe significância suficiente para afirmar que a média de colesterol na população antes da dieta e depois
# da dieta são diferentes, e podemos observar que a média de colesterol na população após a dieta é menor que antes da dieta.


# # EXEMPLO 6 (SCORE DE DESEMPENHO DE EMPRESAS)

piedpiper = np.array(
    [
        4.57,
        4.55,
        5.47,
        4.67,
        5.41,
        5.55,
        5.53,
        5.63,
        3.86,
        3.97,
        5.44,
        3.93,
        5.31,
        5.17,
        4.39,
        4.28,
        5.25,
    ]
)
endframe = np.array(
    [
        4.27,
        3.93,
        4.01,
        4.07,
        3.87,
        4.0,
        4.0,
        3.72,
        4.16,
        4.1,
        3.9,
        3.97,
        4.08,
        3.96,
        3.96,
        3.77,
        4.09,
    ]
)


check_normality(piedpiper)
check_normality(endframe)


# APLICANDO WILCOXON SIGNED RANK (NÃO PARAMÉTRICO) PARA AMOSTRAS PAREADAS

# Hipóteses
# H0: as médias são iguais
# H1: as médias são diferentes

# O Wilcoxon Signed Rank é um teste não paramétrico que compara dois conjuntos de dados pareados ou dependentes, similar ao t-test para amostras pareadas.
# Uso: Quando você tem duas amostras dependentes/pareadas e os dados não seguem uma distribuição normal.
# Suposição: Não exige normalidade, mas os dados devem ser pareados (ex.: medições antes e depois de um tratamento).

# Conclusões:
# Se pvalue <= 0.05, rejeita H0 e assume H1;
# Se pvalue > 0.05 não deve rejeitar H0.

test, pvalue = stats.wilcoxon(endframe, piedpiper)  # alternative default two sided
print(f"p-value: {pvalue :.6f} >> one_tailed_pval: {pvalue / 2 :.6f}")

test, one_sided_pvalue = stats.wilcoxon(endframe, piedpiper, alternative="less")
print(f"one sided pvalue: {one_sided_pvalue :.6f}")
if pvalue < 0.05:
    print("Reject null hypothesis")
else:
    print("Fail to recejt null hypothesis")

# Conclusção: Existe significância suficiente para afirmar que a média do score na população da empresa endframe e piedpiper
# são diferentes, e podemos observar que a média do score da empresa piedpiper é maior que o da empresa endframe.
