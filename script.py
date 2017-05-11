import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())


alive = train.Survived
sex = train.Sex
s = []
first = []

#set up matrix of input variables
for i in range(len(sex)):
    if sex[i] == 'female':
        temp = 1
    elif sex[i] == 'male':
        temp = 0
    s.append(temp)
    first.append(1)

x = pd.DataFrame(first)
x[['Age','Fare']] = train[['Age','Fare']]
x['Sex'] = s

#Check the data for any obvious clustering
fig, ax = plt.subplots()
train.plot(kind="scatter", x="Age", y="Fare", s=50, 
                            c="Survived", ax=ax, cmap="plasma")
plt.savefig('fig1')

fig, ax = plt.subplots()
x[['Sex','Fare']].plot(kind="scatter", x="Sex", y="Fare", s=50, 
                            c=train.Survived, ax=ax, cmap="plasma")
plt.savefig('fig2')

fig, ax = plt.subplots()
train[['Age','Survived']].plot(kind="scatter", x="Age", y="Survived", s=50, 
                            c=x.Sex, ax=ax, cmap="plasma")
                            
plt.savefig('fig3')

fig, ax = plt.subplots()

x.plot(kind='scatter', x='Age', y='Fare', c= 'Sex', ax=ax, cmap='plasma')

plt.savefig('fig4')


#Find and replace all the null values with the median value
aNull = x.Age.isnull()
meanAge = round(x.Age.sum()/len(x.Age),0)

for i in range(len(aNull)):
    if aNull[i] == True:
        x.Age[i] = meanAge

#----------------------Multivariable linear regression method--------------------------- 
xtx = np.dot(x.transpose(),x)
xty = np.dot(x.transpose(),alive)
inverse = np.linalg.inv(xtx)

b = np.dot(xty,inverse)

h = np.dot(x,b.transpose())
predicted = []

for i in range(len(h)):
    if h[i] >=0.50:
        predicted.append(1)
    else:
        predicted.append(0)

#Check error in calculations (percent error,and R^2)
error = (abs(predicted-alive)).sum()/len(predicted)
mean = alive.sum()/len(alive)
tot = ((alive-mean)**2).sum()
res = ((alive-predicted)**2).sum()
R2 = 1-res/tot

print ('LR: %i,%i' %(error*100, R2))
alivePredicted = pd.DataFrame({'Predicted': predicted, 'Survived': alive})


#-----------------------run algorithm on test data-----------------------------------
#Set everything up in a convienent dataFrame
sex = test.Sex
s = []
first = []

#set up matrix of input variables
for i in range(len(sex)):
    if sex[i] == 'female':
        temp = 1
    elif sex[i] == 'male':
        temp = 0
    s.append(temp)
    first.append(1)

x1 = pd.DataFrame(first)
x1[['Age','Fare']] = test[['Age','Fare']]
x1['Sex'] = s

#Find and replace all the null values with the median value
aNull = x1.Age.isnull()
fNull = x1.Fare.isnull()
meanFare = round(x1.Fare.sum()/len(x1.Fare),2)
meanAge = round(x1.Age.sum()/len(x1.Age),0)

for i in range(len(aNull)):
    if aNull[i] == True:
        x1.Age[i] = meanAge

for i in range(len(fNull)):
    if fNull[i] == True:
        x1.Fare[i] = meanFare

h = np.dot(x1,b.transpose())

predicted = []
for i in range(len(h)):
    if h[i] >=0.50:
        predicted.append(1)
    else:
        predicted.append(0)

testLR = pd.DataFrame(test)

testLR.insert(1,'Survived', predicted)


#-------------------------------KNN approach--------------------------------------------------

#Knn algorithm


#Split the data into seperate groups based on gender and age groups
oldBoys = x.loc[ (x.Age >= 60) 
                & (x.Sex == 0)].sort_values('Fare', ascending = False)
lessOldBoys = x.loc[(x.Age >= 40) & (x.Age < 60) 
                & (x.Sex ==0)].sort_values('Fare', ascending = False)
midBoys = x.loc[(x.Age >= 20) & (x.Age < 40) 
                & (x.Sex ==0)].sort_values('Fare', ascending = False)
youngBoys = x.loc[(x.Age >= 0) & (x.Age < 20) 
                & (x.Sex ==0)].sort_values('Fare', ascending = False)
   
oldGirls = x.loc[ (x.Age >= 60) 
                & (x.Sex == 1)].sort_values('Fare', ascending = False)
lessOldGirls = x.loc[(x.Age >= 40) & (x.Age < 60) 
                & (x.Sex ==1)].sort_values('Fare', ascending = False)
midGirls = x.loc[(x.Age >= 20) & (x.Age < 40) 
                & (x.Sex ==1)].sort_values('Fare', ascending = False)
youngGirls = x.loc[(x.Age >= 0) & (x.Age < 20)
                & (x.Sex ==1)].sort_values('Fare', ascending = False)

k = 21

y=0
count = 1
remain = len(oldBoys) % k
low =[]
high = []
guess = []
lcount = 0

#Group k nearby variables based on fare for the various age and gender groups
for i in oldBoys.index:
    
    #do the knn algorithm for the remainder
    if count > (len(oldBoys)-remain):
        if (count % k - 1) == 0:
            ave = (x.Fare[x.Fare.index == i].iloc[0] + low[lcount-1])/2
            high.append(ave)
            low[lcount-1] = ave
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        if count == len(oldBoys):
            low.append(0)
            g = (1/remain)*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
                
    #split the categories into groups of k=5
    else:
        if count == 1 or (count % k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0]+low[lcount-1])/2
                low[lcount-1] = ave
                high.append(ave)
            else:
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        
        if count % k == 0:
            if low:
                low.append(ave)
            else:
                low.append(x.Fare[x.Fare.index == i].iloc[0])
            lcount += 1
            g = 1/k*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
            y = 0
            if count == len(oldBoys):
                low[lcount-1] = 0
    count+=1 

h1 = pd.DataFrame({'Survived': guess,'Age': 60,'Sex': 0,
                    'Fare Low': low, 'Fare High': high})



y=0
count = 1
remain = len(lessOldBoys)%k
low =[]
high = []
guess = []
lcount = 0

for i in lessOldBoys.index:
    
    #do the knn algorithm for the remainder
    if count > (len(lessOldBoys)-remain):
        if (count %k - 1) == 0:
            ave = (x.Fare[x.Fare.index == i].iloc[0] + low[lcount-1])/2
            high.append(ave)
            low[lcount-1] = ave
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        if count == len(lessOldBoys):
            low.append(0)
            lcount += 1
            g = 1/remain*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
                
    #split the categories into groups of k=5
    else:
        if count == 1 or (count %k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0]+low[lcount-1])/2
                low[lcount-1] = ave
                high.append(ave)
            else:
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        
        if count % k == 0:
            if low:
                low.append(ave)
            else:
                low.append(x.Fare[x.Fare.index == i].iloc[0])
            lcount += 1
            g = 1/k*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
            y = 0
            if count == len(lessOldBoys):
                low[lcount-1] = 0
    count+=1 

h2 = pd.DataFrame({'Survived': guess,'Age': 40,'Sex': 0,
                    'Fare Low': low, 'Fare High': high})

y=0
count = 1
remain = len(midBoys)%k
low =[]
high = []
guess = []
lcount = 0

for i in midBoys.index:
    
    #do the knn algorithm for the remainder
    if count > (len(midBoys)-remain):
        if (count %k - 1) == 0:
            ave = (x.Fare[x.Fare.index == i].iloc[0] + low[lcount-1])/2
            high.append(ave)
            low[lcount-1] = ave
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        if count == len(midBoys):
            low.append(0)
            lcount += 1
            g = 1/remain*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
                
    #split the categories into groups of k=5
    else:
        if count == 1 or (count %k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0]+low[lcount-1])/2
                low[lcount-1] = ave
                high.append(ave)
            else:
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        
        if count % k == 0:
            if low:
                low.append(ave)
            else:
                low.append(x.Fare[x.Fare.index == i].iloc[0])
            lcount += 1
            g = 1/k*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
            y = 0
            if count == len(midBoys):
                low[lcount-1] = 0
    count+=1 

h3 = pd.DataFrame({'Survived': guess,'Age': 20,'Sex': 0,
                    'Fare Low': low, 'Fare High': high})

y=0
count = 1
remain = len(youngBoys)%k
low =[]
high = []
guess = []
lcount = 0

for i in youngBoys.index:
    
    #do the knn algorithm for the remainder
    if count > (len(youngBoys)-remain):
        if (count %k - 1) == 0:
            ave = (x.Fare[x.Fare.index == i].iloc[0] + low[lcount-1])/2
            high.append(ave)
            low[lcount-1] = ave
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        if count == len(youngBoys):
            low.append(0)
            lcount += 1
            g = 1/remain*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
                
    #split the categories into groups of k=5
    else:
        if count == 1 or (count %k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0]+low[lcount-1])/2
                low[lcount-1] = ave
                high.append(ave)
            else:
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        
        if count % k == 0:
            if low:
                low.append(ave)
            else:
                low.append(x.Fare[x.Fare.index == i].iloc[0])
            lcount += 1
            g = 1/k*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
            y = 0
            if count == len(youngBoys):
                low[lcount-1] = 0
    count+=1 

h4 = pd.DataFrame({'Survived': guess,'Age': 0,'Sex': 0,
                    'Fare Low': low, 'Fare High': high})

y=0
count = 1
remain = len(oldGirls)%k
low =[]
high = []
guess = []
lcount = 0

for i in oldGirls.index:
    
    #do the knn algorithm for the remainder
    if count > (len(oldGirls)-remain):
        if (count %k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0] + low[lcount-1])/2
                high.append(ave)
                low[lcount-1] = ave
            else: 
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        if count == len(oldGirls):
            low.append(0)
            lcount += 1
            g = 1/remain*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
                
    #split the categories into groups of k=5
    else:
        if count == 1 or (count %k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0]+low[lcount-1])/2
                low[lcount-1] = ave
                high.append(ave)
            else:
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        
        if count % k == 0:
            if low:
                low.append(ave)
            else:
                low.append(x.Fare[x.Fare.index == i].iloc[0])
            lcount += 1
            g = 1/k*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
            y = 0
            if count == len(oldGirls):
                low[lcount-1] = 0
    count+=1 

h5 = pd.DataFrame({'Survived': guess,'Age': 60,'Sex': 1,
                    'Fare Low': low, 'Fare High': high})

y=0
count = 1
remain = len(lessOldGirls)%k
low =[]
high = []
guess = []
lcount = 0

for i in lessOldGirls.index:
    
    #do the knn algorithm for the remainder
    if count > (len(lessOldGirls)-remain):
        if (count %k - 1) == 0:
            ave = (x.Fare[x.Fare.index == i].iloc[0] + low[lcount-1])/2
            high.append(ave)
            low[lcount-1] = ave
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        if count == len(lessOldGirls):
            low.append(0)
            lcount += 1
            g = 1/remain*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
                
    #split the categories into groups of k=5
    else:
        if count == 1 or (count %k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0]+low[lcount-1])/2
                low[lcount-1] = ave
                high.append(ave)
            else:
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        
        if count % k == 0:
            if low:
                low.append(ave)
            else:
                low.append(x.Fare[x.Fare.index == i].iloc[0])
            lcount += 1
            g = 1/k*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
            y = 0
            if count == len(lessOldGirls):
                low[lcount-1] = 0
    count+=1 

h6 = pd.DataFrame({'Survived': guess,'Age': 40,'Sex': 1,
                    'Fare Low': low, 'Fare High': high})

y=0
count = 1
remain = len(midGirls)%k
low =[]
high = []
guess = []
lcount = 0

for i in midGirls.index:
    
    #do the knn algorithm for the remainder
    if count > (len(midGirls)-remain):
        if (count %k - 1) == 0:
            ave = (x.Fare[x.Fare.index == i].iloc[0] + low[lcount-1])/2
            high.append(ave)
            low[lcount-1] = ave
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        if count == len(midGirls):
            low.append(0)
            lcount += 1
            g = 1/remain*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
                
    #split the categories into groups of k=5
    else:
        if count == 1 or (count %k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0]+low[lcount-1])/2
                low[lcount-1] = ave
                high.append(ave)
            else:
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        
        if count % k == 0:
            if low:
                low.append(ave)
            else:
                low.append(x.Fare[x.Fare.index == i].iloc[0])
            lcount += 1
            g = 1/k*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
            y = 0
            if count == len(midGirls):
                low[lcount-1] = 0
    count+=1 

h7 = pd.DataFrame({'Survived': guess,'Age': 20,'Sex': 1,
                    'Fare Low': low, 'Fare High': high})

y=0
count = 1
remain = len(youngGirls)%k
low =[]
high = []
guess = []
lcount = 0

for i in youngGirls.index:
    
    #do the knn algorithm for the remainder
    if count > (len(youngGirls)-remain):
        if (count %k - 1) == 0:
            ave = (x.Fare[x.Fare.index == i].iloc[0] + low[lcount-1])/2
            high.append(ave)
            low[lcount-1] = ave
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        if count == len(youngGirls):
            low.append(0)
            lcount += 1
            g = 1/remain*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
                
    #split the categories into groups of k=5
    else:
        if count == 1 or (count %k - 1) == 0:
            if high:
                ave = (x.Fare[x.Fare.index == i].iloc[0]+low[lcount-1])/2
                low[lcount-1] = ave
                high.append(ave)
            else:
                high.append(x.Fare[x.Fare.index == i].iloc[0])
        y += train.loc[train.index == i, 'Survived'].iloc[0]
        
        if count % k == 0:
            if low:
                low.append(ave)
            else:
                low.append(x.Fare[x.Fare.index == i].iloc[0])
            lcount += 1
            g = 1/k*(y)
            if g >= 0.5:
                guess.append(1)
            else:
                guess.append(0)
            y = 0
            
        if count == len(youngGirls):
            low[lcount-1] = 0
    count+=1 

h8 = pd.DataFrame({'Survived': guess,'Age': 0,'Sex': 1,
                    'Fare Low': low, 'Fare High': high})

#Combine results of the knn approach to the 8 seperate groups
h = pd.concat([h1,h2,h3,h4,h5,h6,h7,h8])

#----------------------------------------Find Error in KNN------------------------

prediction =[]

#Search the prediction matrix h for a valuem of survived that matches the input
#parameters of the individual
for i in range(len(x)):
    if x.Age.loc[x.Age.index == i].iloc[0] < 80:
        prediction.append(h.Survived.loc[(x.Age.loc[x.Age.index == i].iloc[0] >= h.Age) 
            & (x.Age.loc[x.Age.index == i].iloc[0] < (h.Age +20))
            & (x.Fare.loc[x.Fare.index == i].iloc[0] >= h['Fare Low'])
            & (x.Fare.loc[x.Fare.index == i].iloc[0] <= h['Fare High'])
            & (x.Sex.loc[x.Sex.index == i].iloc[0] == h.Sex)].iloc[0])
    elif x.Age.loc[x.Age.index == i].iloc[0] >= 80:
        prediction.append(h.Survived.loc[(x.Age.loc[x.Age.index == i].iloc[0] >= h.Age)
            & (x.Age.loc[x.Age.index == i].iloc[0] <= (h.Age +20))
            & (x.Fare.loc[x.Fare.index == i].iloc[0] >= h['Fare Low'])
            & (x.Fare.loc[x.Fare.index == i].iloc[0] <= h['Fare High'])
            & (x.Sex.loc[x.Sex.index == i].iloc[0] == h.Sex)].iloc[0])

#Calculate error
error = (abs(prediction-alive)).sum()/len(prediction)
mean = alive.sum()/len(alive)
tot = ((alive-mean)**2).sum()
res = ((alive-prediction)**2).sum()
R2 = 1-res/tot

print ('KNN: %i,%i' %(error*100, R2))

#-------------------------------Test KNN on test data------------------------------

prediction =[]

#Search the prediction matrix h for a valuem of survived that matches the input
#parameters of the individual
x1 = x1.drop(0,1)
for i in range(len(x1)):
    if x1.Age.loc[x1.Age.index == i].iloc[0] < 80:
        topMoney = h['Fare High'].loc[(x1.Age.loc[x1.Age.index == i].iloc[0] >= h.Age) 
                                        & (x1.Age.loc[x1.Age.index == i].iloc[0] < (h.Age +20))
                                        & (h.Sex == x1.Sex.loc[x1.Sex.index == i].iloc[0])].iloc[0]
        if x1.Fare.loc[x1.Fare.index == i].iloc[0] <= topMoney:
            prediction.append(h.Survived.loc[(x1.Age.loc[x1.Age.index == i].iloc[0] >= h.Age) 
                        & (x1.Age.loc[x1.Age.index == i].iloc[0] < (h.Age +20))
                        & (x1.Sex.loc[x1.Sex.index == i].iloc[0] == h.Sex)
                        & (x1.Fare.loc[x1.Fare.index == i].iloc[0] >= h['Fare Low'])
                        & (x1.Fare.loc[x1.Fare.index == i].iloc[0] <= h['Fare High'])].iloc[0])
        else: 
            prediction.append(h.Survived.loc[(x1.Age.loc[x1.Age.index == i].iloc[0] >= h.Age) 
                        & (x1.Age.loc[x1.Age.index == i].iloc[0] <= (h.Age +20))
                        & (x1.Sex.loc[x1.Sex.index == i].iloc[0] == h.Sex)
                        & (x1.Fare.loc[x1.Fare.index == i].iloc[0] >= h['Fare Low'])].iloc[0])
        
           
    elif x1.Age.loc[x1.Age.index == i].iloc[0] >= 80:
        predict = h.loc[(x1.Age.loc[x1.Age.index == i].iloc[0] >= h.Age) 
                        & (x1.Age.loc[x1.Age.index == i].iloc[0] <= (h.Age +20))].iloc[0]
        topMoney = h['Fare High'].loc[(x1.Age.loc[x1.Age.index == i].iloc[0] >= h.Age) 
                                        & (x1.Age.loc[x1.Age.index == i].iloc[0] <= (h.Age +20))
                                        & (h.Sex == x1.Sex.loc[x1.Sex.index == i].iloc[0])].iloc[0]
        if x1.Fare.loc[x1.Fare.index == i].iloc[0] <= topMoney:
            predicition.append(h.Survived.loc[(x1.Age.loc[x1.Age.index == i].iloc[0] >= h.Age) 
                        & (x1.Age.loc[x1.Age.index == i].iloc[0] <= (h.Age +20))
                        & (x1.Sex.loc[x1.Sex.index == i].iloc[0] == h.Sex)
                        & (x1.Fare.loc[x1.Fare.index == i].iloc[0] >= h['Fare Low'])
                        & (x1.Fare.loc[x1.Fare.index == i].iloc[0] <= h['Fare High'])].iloc[0])
        else: 
            predicition.append(h.Survived.loc[(x1.Age.loc[x1.Age.index == i].iloc[0] >= h.Age) 
                        & (x1.Age.loc[x1.Age.index == i].iloc[0] <= (h.Age +20))
                        & (x1.Sex.loc[x1.Sex.index == i].iloc[0] == h.Sex)
                        & (x1.Fare.loc[x1.Fare.index == i].iloc[0] >= h['Fare Low'])].iloc[0])
        
testKNN = pd.DataFrame(test.drop('Survived',1))
testKNN.insert(1,'Survived',prediction)

#Any files you save will be available in the output tab below
testKNN.to_csv('testKNN.csv')
testLR.to_csv('testLR.csv')
