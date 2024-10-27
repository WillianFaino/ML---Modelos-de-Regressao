import time as t
import random as r
import numpy as np
import pandas as pd
import seaborn as sb
from math import sqrt
import matplotlib as plt
from sklearn.svm import SVR
from statistics import mean
from statistics import pstdev
from tabulate import tabulate
import statsmodels.formula.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




dados = pd.read_csv('C:\insurance.csv')

trainingSet, testingSet = \
    train_test_split(dados, test_size=0.30, \
                      random_state=(int)(r.random()*99999))

# trainingSet.to_excel('Bilonga.xlsx')
#
# testingSet.to_excel('Bilango.xlsx')


# print(dados.shape)
# dados


age = dados.iloc[:,0].values
sex = dados.iloc[:,1].values
bmi = dados.iloc[:,2].values
children = dados.iloc[:,3].values
smoker = dados.iloc[:,4].values
region = dados.iloc[:,5].values
charges = dados.iloc[:,6].values

# print(age)
# print(sex)
# print(bmi)
# print(children)
# print(smoker)
# print(region)
# print(charges)

def strToBool(str):
    strBool = []


    for i in range(len(str)):
        if str[i]=="male":
            strBool.append(False)


        elif str[i]=="female":
            strBool.append(True)

        elif str[i] == "no":
            strBool.append(False)


        elif str[i] == "yes":
            strBool.append(True)

    return strBool

def strToInt(str):
    strInt = []


    for i in range(len(str)):
        if str[i]=="southwest":
            strInt.append(3)

        elif str[i]=="southeast":
            strInt.append(2)

        elif str[i] == "northwest":
            strInt.append(1)

        elif str[i] == "northeast":
            strInt.append(0)

    return strInt





corChar = []

corAgeChar = np.corrcoef(age, charges)
corSexChar = np.corrcoef(strToBool(sex), charges)
corBmiChar = np.corrcoef(bmi, charges)
corChildrenChar = np.corrcoef(children, charges)
corSmokerChar = np.corrcoef(strToBool(smoker), charges)
corRegionChar = np.corrcoef(strToInt(region), charges)



corChar.append(corAgeChar[0][1])
corChar.append(corSexChar[0][1])
corChar.append(corBmiChar[0][1])
corChar.append(corChildrenChar[0][1])
corChar.append(corSmokerChar[0][1])
corChar.append(corRegionChar[0][1])

detChar = []

for i in range(len(corChar)):
    r2 = corChar[i]**2
    umr2 = 1-r2
    restodaequacao = ((dados.size/7)-1)/((dados.size/7)-(len(corChar)+1))
    restodoresto = umr2*restodaequacao


    detChar.append(1-restodoresto)




# print("\n\n\n\ncor. age-charges: ", corChar[0])
# print("cor. sex-charges: ", corChar[1])
# print("cor. bmi-charges: ", corChar[2])
# print("cor. children-charges: ", corChar[3])
# print("cor. smoker-charges: ", corChar[4])
# print("cor. region-charges: ", corChar[5])



corAll = np.corrcoef([age,strToBool(sex),bmi,children,strToBool(smoker),strToInt(region)])





cabeçadores = ["","age", "sex", "bmi", "children", "smoker", "region"]
corAllL = corAll.tolist()




for i in range(len(corAllL)):
    corAllL[i].insert(0,cabeçadores[i+1])



# print(tabulate(corAllL,headers=cabeçadores))

detAllL = []

for i in range(len(corAll)):
    detAllL.append([])
    for j in range(len(corAll)):
        r2 = corAll[i][j]**2
        umr2 = 1-r2
        restodaequacao = ((dados.size/7)-1)/((dados.size/7)-(len(corAll)+1))
        restodoresto = umr2*restodaequacao


        detAllL[i].append(1-restodoresto)




for i in range(len(detAllL)):
    detAllL[i].insert(0,cabeçadores[i+1])


# print(tabulate(detAllL,headers=cabeçadores))


# print(detChar)
lista_to_excel = []
lista_to_excel.append(cabeçadores)
corChar.insert(0,'correlação')
lista_to_excel.append(corChar)

lista_to_excel.append([])
lista_to_excel.append([])
lista_to_excel.append([])


lista_to_excel.append(cabeçadores)
for i in range(len(corAllL)):
    lista_to_excel.append(corAllL[i])

lista_to_excel.append([])
lista_to_excel.append([])
lista_to_excel.append([])


lista_to_excel.append(cabeçadores)
detChar.insert(0,'determinação')
lista_to_excel.append(detChar)

lista_to_excel.append([])
lista_to_excel.append([])
lista_to_excel.append([])

lista_to_excel.append(cabeçadores)
for i in range(len(detAllL)):
    lista_to_excel.append(detAllL[i])




df = pd.DataFrame(lista_to_excel)
df.to_excel("dados_pro_relatorio_ana_interprete.xlsx")




# print("cor. age - charges:\n",corAgeChar)
# print("cor. sex - charges:\n",corSexChar)
# print("cor. bmi - charges:\n",corBmiChar)
# print("cor. children - charges:\n",corChildrenChar)
# print("cor. smoker - charges:\n",corSmokerChar)
# print("cor. region - charges:\n",corRegionChar)



#REGRESSÕES

                                    #####MULTIPLE LINEAR REGRESSION#####

initT = t.time()
MLineReg_smse_mean = 0
MLineReg_smse_allVal = []
MLineReg_StDev = 0

for i in range(20):
    trainingSet, testingSet = \
        train_test_split(dados, test_size=0.30, \
                         random_state=(int)(r.random() * 99999))

    FModelAdj = sm.ols(formula="charges ~ age + bmi + children + smoker", data=trainingSet)
    FModelTrain = FModelAdj.fit()

    reg = FModelTrain.predict(testingSet)

    mlrsmse = sqrt(mean_squared_error(reg, testingSet.iloc[:,6].values))
    MLineReg_smse_allVal.append(mlrsmse)
    MLineReg_smse_mean+=mlrsmse
    print("End of execution #", i)
finalT = t.time()-initT
fm = int(int(finalT)/60)
fs = int(finalT)%60
fms = int(finalT*1000)%1000
print("Mutiple Linear Regression execution time = ", fm,"m",fs,"s",fms,"ms")



MLineReg_smse_mean = MLineReg_smse_mean/20
print("Square Root of mean squared error (Multiple Linear Regression) = ", MLineReg_smse_mean)
MLineReg_StDev = pstdev(MLineReg_smse_allVal)
print("Standard Deviation (Multiple Linear Regression) = ", MLineReg_StDev)




# dadosAdj = dados.drop("sex", axis=1)
# dadosAdj = dadosAdj.drop("region", axis=1)   ##removal of the least impactant attributes
#
# for i in range(len(dadosAdj.iloc[:,3])):
#     if dadosAdj.iloc[i,3] == "yes":
#         dadosAdj.iloc[i,3] = True
#
#     elif dadosAdj.iloc[i,3] == "no":
#         dadosAdj.iloc[i,3] = False
#
#
# attSet = dadosAdj.drop("charges", axis=1)
# chargesSet = dadosAdj["charges"]

# MLPReg_smse_mean = 0
# MLPReg_smse_allVal = []
# MLPReg_StDev = 0
#
#
#
#                                     #####MLP REGRESSION#####
# initT = t.time()
# for i in range(20):
#     trainingSet, testingSet, trainingC, testingC = \
#         train_test_split(attSet, chargesSet, test_size=0.30, \
#                          random_state=(int)(r.random() * 99999))
#
#
#     MLPRegAdj = MLPRegressor(hidden_layer_sizes=(32,32,32,32,32,32,32,32),max_iter=500)
#     MLPRegTrain = MLPRegAdj.fit(trainingSet, trainingC)
#
#
#     reg = MLPRegTrain.predict(testingSet)
#
#     mlpregsmse = sqrt(mean_squared_error(reg, testingC.values))
#     MLPReg_smse_allVal.append(mlpregsmse)
#
#     MLPReg_smse_mean+=mlpregsmse
#     print("End of execution #",i)
#
#
# finalT = t.time()-initT
# fm = int(int(finalT)/60)
# fs = int(finalT)%60
# fms = int(finalT*1000)%1000
# print("MLP Regression execution time = ", fm,"m",fs,"s",fms,"ms")
#
#
# MLPReg_smse_mean = MLPReg_smse_mean/20
# print("\n\n\nSquare Root of mean squared error (MultiLayer Perceptron Regression) = ", MLPReg_smse_mean)
# MLPReg_StDev = pstdev(MLPReg_smse_allVal)
# print("Standard Deviation (MultiLayer Perceptron Regression) = ", MLPReg_StDev)





# svr_smse_mean = 0
# svr_smse_allVal = []
# svr_StDev = 0
#
#
#                                     ##### SVR #####
# initT = t.time()
# for i in range(20):
#     trainingSet, testingSet, trainingC, testingC = \
#         train_test_split(attSet, chargesSet, test_size=0.30, \
#                          random_state=(int)(r.random() * 99999))
#
#
#     svreg = SVR(C = 100000.0, kernel="poly")
#     svregTrain= svreg.fit(trainingSet, trainingC)
#
#
#     reg = svregTrain.predict(testingSet)
#
#     svrsmse = sqrt(mean_squared_error(reg, testingC.values))
#     svr_smse_allVal.append(svrsmse)
#
#     svr_smse_mean+=svrsmse
#     print("End of execution #",i)
#
#
# finalT = t.time()-initT
# fm = int(int(finalT)/60)
# fs = int(finalT)%60
# fms = int(finalT*1000)%1000
# print("SVR execution time = ", fm,"m",fs,"s",fms,"ms")
#
#
# svr_smse_mean = svr_smse_mean/20
# print("\n\n\nSquare Root of mean squared error (SVR) = ", svr_smse_mean)
# svr_StDev = pstdev(svr_smse_allVal)
# print("Standard Deviation (SVR) = ", svr_StDev)







# dtr_smse_mean = 0
# dtr_smse_allVal = []
# dtr_StDev = 0
#
#
#                                     ##### DT Regression #####
# initT = t.time()
# for i in range(20):
#     trainingSet, testingSet, trainingC, testingC = \
#         train_test_split(attSet, chargesSet, test_size=0.30, \
#                          random_state=(int)(r.random() * 99999))
#
#
#     dtreg = DecisionTreeRegressor(max_leaf_nodes=100)
#     dtregTrain= dtreg.fit(trainingSet, trainingC)
#
#
#     reg = dtregTrain.predict(testingSet)
#
#     dtrsmse = sqrt(mean_squared_error(reg, testingC.values))
#     dtr_smse_allVal.append(dtrsmse)
#
#     dtr_smse_mean+=dtrsmse
#     print("End of execution #",i)
#     print("leaf nodes = ", dtreg.get_n_leaves())
#
#
# finalT = t.time()-initT
# fm = int(int(finalT)/60)
# fs = int(finalT)%60
# fms = int(finalT*1000)%1000
# print("DT Regression execution time = ", fm,"m",fs,"s",fms,"ms")
#
#
# dtr_smse_mean = dtr_smse_mean/20
# print("\n\n\nSquare Root of mean squared error (DT Regression) = ", dtr_smse_mean)
# dtr_StDev = pstdev(dtr_smse_allVal)
# print("Standard Deviation (DT Regression) = ", dtr_StDev)




























