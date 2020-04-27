'''بسم الله الرحمن الرحيم'''               

                                    #Project 5 (Employee Psitions and salaries) using sheet excel of real employees
##ANOTHER PROJECT FOR POLYNOMIAL REGRESSION
#Import all liberaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Import Sklearn with the function (train_test_split)
from sklearn.model_selection import train_test_split
#Import LinearRegression
from sklearn.linear_model import LinearRegression
#Import the file sheet
data=pd.read_csv(r"H:\NEW PARTITION (F)\deploma\PYTHON\sample sheets\Position_Salaries.csv")
#Assign the input "level(of the postition title)" for X
X=data["Level"]
print(X)
#Import the output(that i will predict) for Salary As [Y]
Y=data["Salary"]
print(Y)
#reshape X & Y to an (-1,1) array(جرب الكود من غيرها وهتفهم لية عملناها وشوف الخطا)
x=np.array(X).reshape(-1,1)
y=np.array(Y).reshape(-1,1)
#دلوقتى انا ظهر ليا الpattern بس ماينفعش معاها الlinearRegression عشان الpattern مش مظبوطة
#import the function of PolynomialFunction from Sklearn liberary
#the  math liberary of (sklearn.preprocessing & the function of PolynomialFeatures) is used to make a function of degrees to predict optimum line solution for the pattern 
from sklearn.preprocessing import PolynomialFeatures
#دلوقتى انا همشى بالمعادلة بتاعى الpolynomial اللى هيا فيها درجات للاسس من معادلات من الدرجة الثانية او الثالثة او اى درجة (ابحث بجوجل)
'''poly_obj=PolynomialFeatures(degree=2)'
'poly_obj=PolynomialFeatures(degree=3)'
'poly_obj=PolynomialFeatures(degree=4)'
'poly_obj=PolynomialFeatures(degree=5)'
'poly_obj=PolynomialFeatures(degree=6)'
'poly_obj=PolynomialFeatures(degree=7)'''
#باستخدام المكتبة السابق ذكرها (polynomialfeatures) تقوم بعمل المعادلة الخاصة بالpolynomial وتقوم برفع الX للاوسس (Xpower 2) او زيادة درجة المعادلة(إبحث جوجل)..وبيكون عندة القدرة انة يش على المعادلة يخليهاأوس الرقم اللى محددة فى الdegree
#بعد تجربة الدرجة الثانية لم نجد ان الارقام اللى تم التبوء بها قريبة من الارقام الحقيقية
#فضلنا نجرب للدرجة الثالثة والرابعة والخامسة حتى وصلنا للدرجة السادسة(وبعدها جربت الدرجة السابعة ولاقيت ان الارقام بدات تبعد تانى-فرجعت تانى لافضل تنبوء اللى هوة الoptimum) فى المعادلة اللى هعمل بيها التنبوء وهى كالاتى: 
poly_obj=PolynomialFeatures(degree=6)
#the code to tell the compiler to transform the polynomial frunction to the degree supported
#الامر دة عشان اقول للكومبيلر خش على الpoly_obj واعملها تحويل للدرجة المذكورة
xpoly=poly_obj.fit_transform(x)
print(xpoly)
#Import the Function of (train_test_split) for X & Y And specify the percentage for the prediction
#(random state) to take aspecific raw constant and for not change it every time the compiler make the prediction
X_train,X_test,Y_train,Y_test=train_test_split(xpoly,Y,test_size=0.2,random_state=0)
#to make the linear regression for drawing the represented line for the pattern 
reg=LinearRegression()
#train the level of the careers & their salary
reg.fit(X_train,Y_train)
#to make the prediction for the (X_test) the level of the career
ypred=reg.predict(X_test)
print(ypred)

print(Y_test)
#Draw the pattern for X & Y with the polynomial line prediction
plt.scatter(x,y)
plt.plot(xpoly,reg.predict(xpoly))
plt.show()