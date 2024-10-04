import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp

LIMIT = 0.05

sync1 = np.array([94, 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2, 87.9, 87.9, 93.5, 82.3, 79.3, 78.3, 71.6, 88.6, 74.6, 74.1, 80.6]) # 22
asyncr1 = np.array([77.1, 71.7, 91, 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2]) # 14

# Como saber se as médias são diferentes?
# Seguir o passo a passo abaixo


# # Passo 1:
# # Medir a normalidade
# # Rodar teste de normalidade
# normality_sync1 = stats.shapiro(sync1)
# normality_asyncr1 = stats.shapiro(asyncr1)
# if normality_sync1.pvalue < LIMIT:
#     print("rejeitar h0: ", normality_sync1.pvalue)
# else:
#   print("não rejeitar h0: ", normality_sync1.pvalue)
# if normality_asyncr1.pvalue < LIMIT:
#     print("rejeitar h0: ", normality_asyncr1.pvalue)
#  else:
#   print("não rejeitar h0: ", normality_asyncr1.pvalue)



# # Passo 2:
# # Medir a variança
# # Varianças iguais pois o pValue deu 
# _, p_value_var = stats.levene(sync1, asyncr1)
# print(f"Pvalue Teste de levene: {p_value_var}")
# if p_value_var < LIMIT:
#     print("rejeitar h0: ", p_value_var)
# else:
#   print("não rejeitar h0: ", p_value_var)


# Passo 3:
# Teste que necessita de: Normalidade e variança igual
# Verifica se 2 médias são iguais quando as amostras são normais e variança igual
# Pega as 2 médias e compara
# Compara se as médias são iguais

ttest, p_value_ttest = stats.ttest_ind(sync1, asyncr1)
print("p value: ", p_value_ttest)
print("Desde que a hipotese for de um lado >> use p_value / 2 >> p_value")

if p_value_ttest / 2 < LIMIT:
    print("rejeitar h0: ", p_value_ttest)
else:
    print("não rejeitar h0: ", p_value_ttest)

# Médias das populações são diferentes porque o p_value é menor que 0.05, rejeitamos a hipotese nula e assumimos a hipotese alternativa
# Além de o estudo da média sync1 é maior que o asyncr1 


# 2 tipos de testes:
# - One sided -> usar p_value
# - Two sided -> usar p_value / 2 -> pois two sided são duplicados

#############################################################################
# peso em relacao ao tipo de leite
only_breast=np.array([794.1, 716.9, 993. , 724.7, 760.9, 908.2, 659.3 , 690.8, 768.7, 717.3 , 630.7, 729.5, 
             714.1, 810.3, 583.5, 679.9, 865.1])

only_formula=np.array([ 898.8, 881.2, 940.2, 966.2, 957.5, 1061.7, 1046.2, 980.4, 895.6, 919.7, 1074.1, 952.5, 
              796.3, 859.6, 871.1 , 1047.5, 919.1 , 1160.5, 996.9])

both=np.array([976.4, 656.4, 861.2, 706.8, 718.5, 717.1, 759.8, 894.6, 867.6, 805.6, 765.4, 800.3, 789.9, 875.3, 
      740. , 799.4, 790.3, 795.2 , 823.6, 818.7, 926.8, 791.7, 948.3])

_, p_value_var = stats.levene(only_breast, only_formula)

media_breast = np.mean(only_breast)
media_formula = np.mean(only_formula)
media_both = np.mean(both)

print(media_breast, media_formula, media_both)

F, p_value = stats.f_oneway(only_breast, only_formula, both)

if p_value_ttest < 0.05:
    print("reject null hypotesis")
else:
    print("Fail to reject null hypotesis")

posthoc_df = sp.posthoc_ttest([media_breast, media_formula, media_both], equal_var=True, p_adjust="bonferroni")
print(posthoc_df)

################################################################


test_team=np.array([6.2, 7.1, 1.5, 2,3 , 2, 1.5, 6.1, 2.4, 2.3, 12.4, 1.8, 5.3, 3.1, 9.4, 2.3, 4.1])
developer_team=np.array([2.3, 2.1, 1.4, 2.0, 8.7, 2.2, 3.1, 4.2, 3.6, 2.5, 3.1, 6.2, 12.1, 3.9, 2.2, 1.2 ,3.4])

##############################################################################


youtube = np.array([1913, 1879, 1939, 2146, 2040, 2127, 2122, 2156, 2036, 1974, 1956, 2146, 2151, 1943, 2125])

instagram = np.array([2305., 2355., 2203., 2231., 2185., 2420., 2386., 2410., 2340., 2349., 2241., 2396., 2244., 2267., 2281.])

facebook = np.array([2133., 2522., 2124., 2551., 2293., 2367., 2460., 2311., 2178., 2113., 2048., 2443., 2265., 2095., 2528.])

########################################################################

before_diet=np.array([224, 235, 223, 253, 253, 224, 244, 225, 259, 220, 242, 240, 239, 229, 276, 254, 237, 227])
after_diet=np.array([198, 195, 213, 190, 246, 206, 225, 199, 214, 210, 188, 205, 200, 220, 190, 199, 191, 218])

##########################################################################

before_diet=np.array([224, 235, 223, 253, 253, 224, 244, 225, 259, 220, 242, 240, 239, 229, 276, 254, 237, 227])
after_diet=np.array([198, 195, 213, 190, 246, 206, 225, 199, 214, 210, 188, 205, 200, 220, 190, 199, 191, 218])


