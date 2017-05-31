import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, normalize
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile,mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('../input/train.csv')


x = train[['Age','Fare', 'Pclass', 'Parch','SibSp']]
sex = train.Sex
s = []

#set up matrix of input variables
for i in range(len(sex)):
    if sex[i] == 'female':
        temp = 1
    elif sex[i] == 'male':
        temp = 0
    s.append(temp)

x['Sex'] = s
null = x.isnull()
nullNames = []

#Find and replace all null entries in x with the average number
for i in null.sum().index:
    if null.sum().loc[null.sum().index == i].iloc[0] != 0:
        nullNames.append(i)

if nullNames:
    for name in nullNames:
        nullEntries = x[name].isnull()
        meanAge = round(x[name].sum()/len(x[name]),0)

        for i in range(len(nullEntries)):
            if nullEntries[i] == True:
                x[name][i] = meanAge

#Take in the cabin variable and classify each as either 0, or a number corresponding to the letter of each cabin
nullList = train.Cabin.isnull()
cabins = []
for null in nullList:
        if null == True:
                cabins.append(0)
        else:
                cabinType = train.Cabin.loc[train.index==(len(cabins))].iloc[0][0]
                cabins.append(ord(cabinType)-64)


x['Cabin'] = pd.Series(cabins)

#Take in embark and do the same thing as cabin
nullEmbark = train.Embarked.isnull()
embarked = []
for embark in nullEmbark:
        if embark == True:
                embarked.append(0)
        else:
                embarkType = train.Embarked.loc[train.index==(len(embarked))].iloc[0][0]
                embarked.append(ord(embarkType)-64)

x['Embarked'] = pd.Series(embarked)

#Come up with various feature combinations based on their correlations (x.corr()) 
x['Family'] = train['Parch'] + train['SibSp']

'''
x['Pfam'] = x.Family-x.Pclass
x['Pcabin'] = x.Cabin*x.Pclass
x['Fabin'] = x.Fare*x.Cabin
x['Psex'] = x.Parch*x.Sex

x['Farch'] = x.Fare+x.Parch
x['Fembarked'] = x.Fare*x.Embarked
x['Aclass'] = x.Age*x.Pclass

x['Pembark'] = x.Pclass+x.Embarked
x['AgeFam'] =  x.Age-x.Family
x['Fclass'] = x.Fare * x.Pclass
x['FareSib'] = x.Fare*x.SibSp
x['Fsex'] = x.Fare*x.Sex
x['Ffam'] = x.Family*x.Fare


x['SexAgeFam'] = x.Sex * x.AgeFam
'''
x['FarePerFam'] = x.Fare/(x.Family+1)


#Take in all the names and spit out the titles associated with those names (i.e Mr, Mrs. ....) convert to ints
names = train.Name
titleTemp = []
for name in names:
        titleTemp.append(name.split(',')[1].split('.')[0])
titleTitles = []
for til in titleTemp:
        if til not in titleTitles:
                titleTitles.append(til)
title = []
for i in range(len(titleTemp)):
        title.append(titleTitles.index(titleTemp[i])+1)

x['title'] = title

alive = train.Survived

#Add polynomial terms
poly = PolynomialFeatures(4,include_bias=False )
sq = poly.fit_transform(x)

polyDF = pd.DataFrame(sq)

x = pd.concat([x,polyDF], axis=1, join_axes=[x.index])

#Remove the lower percentile of useful features
sel = SelectPercentile(mutual_info_classif,percentile= 10)
sel.fit(x,alive)
x = pd.DataFrame(sel.transform(x))

#Create data to train on and to cross validate on
x_train,x_split,y_train,y_split = train_test_split(x,alive, test_size = 0.1)
#x_cv,x_test,y_cv,y_test = train_test_split(x_split,y_split, test_size = 0.5)

x_cv = x_split
y_cv = y_split

#x_train = x
#y_train = alive

#Scale the data
sclr = StandardScaler()
sclr.fit(x_train)
x_train = pd.DataFrame(sclr.transform(x_train),index=x_train.index)
x_cv = pd.DataFrame(sclr.transform(x_cv),index=x_cv.index)
#x_test = pd.DataFrame(sclr.transform(x_test),index=x_test.index)

#normalize the data
norm = normalize(x_train)
x_train = pd.DataFrame(norm,index=x_train.index)

norm = normalize(x_cv)
x_cv = pd.DataFrame(norm,index=x_cv.index)

#Use a random forest algorithm  to classify the data
clf = RandomForestClassifier(max_depth = 6, n_estimators=15, min_samples_split=6)
clf.fit(x_train,y_train)

print (clf.score(x_cv,y_cv))

#------------------------------Run on test case---------------------------------------

test = pd.read_csv('../input/test.csv')
train = []

x = test[['Age','Fare', 'Pclass', 'Parch','SibSp']]
sex = test.Sex
s = []

#set up matrix of input variables
for i in range(len(sex)):
    if sex[i] == 'female':
        temp = 1
    elif sex[i] == 'male':
        temp = 0
    s.append(temp)

x['Sex'] = s
null = x.isnull()
nullNames = []

#Find and replace all null entries in x with the average number
for i in null.sum().index:
    if null.sum().loc[null.sum().index == i].iloc[0] != 0:
        nullNames.append(i)
print (nullNames)
if nullNames:
    for name in nullNames:
        nullEntries = x[name].isnull()
        meanAge = round(x[name].sum()/len(x[name]),0)

        for i in range(len(nullEntries)):
            if nullEntries[i] == True:
                x[name][i] = meanAge

#Take in the cabin variable and classify each as either 0, or a number corresponding to the letter of each cabin
nullList = test.Cabin.isnull()
cabins = []
for null in nullList:
        if null == True:
                cabins.append(0)
        else:
                cabinType = test.Cabin.loc[test.index==(len(cabins))].iloc[0][0]
                cabins.append(ord(cabinType)-64)


x['Cabin'] = pd.Series(cabins)

#Take in embark and do the same thing as cabin
nullEmbark = test.Embarked.isnull()
embarked = []
for embark in nullEmbark:
        if embark == True:
                embarked.append(0)
        else:
                embarkType = test.Embarked.loc[test.index==(len(embarked))].iloc[0][0]
                embarked.append(ord(embarkType)-64)

x['Embarked'] = pd.Series(embarked)

#Come up with various feature combinations based on their correlations (x.corr()) 
x['Family'] = test['Parch'] + test['SibSp']
x['FarePerFam'] = x.Fare/(x.Family+1)


#Take in all the names and spit out the titles associated with those names (i.e Mr, Mrs. ....) convert to ints
names = test.Name
titleTemp = []
for name in names:
        titleTemp.append(name.split(',')[1].split('.')[0])
titleTitles = []
for til in titleTemp:
        if til not in titleTitles:
                titleTitles.append(til)
title = []
for i in range(len(titleTemp)):
        title.append(titleTitles.index(titleTemp[i])+1)

x['title'] = title

#Add polynomial terms
poly = PolynomialFeatures(4,include_bias=False )
sq = poly.fit_transform(x)
polyDF = pd.DataFrame(sq)
x = pd.concat([x,polyDF], axis=1, join_axes=[x.index])

#Scale the data
x = pd.DataFrame(sel.transform(x))
x = pd.DataFrame(sclr.transform(x),index=x.index)

norm = normalize(x)
x = pd.DataFrame(norm,index=x.index)


predict = clf.predict(x)
predict = {'Survived':predict}
testPredict = pd.DataFrame(predict)
testPredict.insert(0,'PassengerId',test['PassengerId'])

testPredict.to_csv('test.csv', index = False)
