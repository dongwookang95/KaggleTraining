import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
'exec(%matplotlib inline)'

data=pd.read_csv('train.csv')
data.head()

data.isnull().sum()

# print(data.head())
# print(data.isnull().sum())
##Age, Cabin and Embarked have null values. I ll try to fix them.


##How many Survived?

f, ax=plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)


ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')

#Type of Features
# Categorical Features = nominal variables /ex) sex 
# Ordinal Features ex) Height -> Tall medium short
# Continous Feature ex) age


#Analysing The Feature
data.groupby(['Sex','Survived'])['Survived'].count()

f, ax=plt.subplots(1,2,figsize=(18,8))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived', data=data,ax=ax[1])
ax[1].set_title('Sex:survived vs dead')
plt.show()

pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='summer_r')

f,ax=plt.subplots(1,2,figsize=(18,8))
data['Pclass'].value_counts().plot.bar(color=['#CF7F32', '#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()
#Above looks interesting. The number of men of the ship is lot more than the number of women.
# Still the number of women saved is almost twice the numbe rof males saved. The survival rates for a women on
# the ship is around 75% while that foor men in around 18-19%
# This looks to be a very important feature for modeling.
# But is it the best? Let s check other features

pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex',data=data)

print('Oldest Passenger was of:', data['Age'].max(), 'Years')
print('Youngest Passenger was of:', data['Age'].min(), 'Years')
print('Average Passenger was of:', data['Age'].mean(), 'Years')

f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age",hue="Survived", data=data, split=True, ax=ax[0])
ax[0].set_title('Pcalss and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))


data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')

pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='summer_r')

data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data.groupby('Initial')['Age'].mean()
print(data.groupby('Initial')['Age'].mean())

## Assigning the NaN Values with the Ceil values of the mean ages
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46

data.Age.isnull().any()

f,ax=plt.subplots(1,2,figsize=(20,10))
data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)

sns.factorplot('Pclass','Survived',col='Initial',data=data)
plt.show()
#Embarked -> Categorical Value
pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Embarked','Survived',data=data)
fig=plt.gcf()
fig.set_size_inches(5,3)

f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=data, ax=ax[0,0])
ax[0,0].set_title('No.Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=data, ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked', hue='Pclass',data=data, ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)
plt.show()

data['Embarked'].fillna('S',inplace=True)
data.Embarked.isnull().any()


# # sibsip->Discrete Feature
# # This feature represents whether a person is alone or with his family members.
# # Sibling = brother,sister,stepbrother,stepsister
# # Spouse = husband,wife

pd.crosstab([data.SibSp],data.Survived).style.background_gradient(cmap='summer_r')

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=data,ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp','Survived',data=data,ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()
# plt.close(2)

# #Parch

pd.crosstab(data.Parch,data.Pclass).style.background_gradient(cmap='summer_r')

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('Parch','Survived',data=data,ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch','Survived',data=data, ax=ax[1], size=6,aspect=1.5)
ax[1].set_title('Parch vs Survived')

plt.close(2)
plt.show()





# # #Fare->Continous Feature..
# # print('Highest Fare was:',data['Fare'].max())
# # print('Lowest Fare was:',data['Fare'].min())
# # print('Average Fare was:',data['Fare'].mean())

# # f, ax=plt.subplots(1,3,figsize=(20,8))
# # sns.distplot(data[data['Pclass']==1].Fare,ax=ax[0])
# # ax[0].set_title('Fares in Pclass 1')
# # sns.distplot(data[data['Pclass']==2].Fare,ax=ax[1])
# # ax[1].set_title('Fares in Pclass 2')
# # sns.distplot(data[data['Pclass']==3].Fare,ax=ax[2])
# # ax[2].set_title('Fares in Pclass 3')



# # sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2)
# # fig=plt.gcf()
# # fig.set_size_inches(10,8)

# # data['Age_band']=0
# # data.loc[data['Age']<=16,'Age_band']=0
# # data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
# # data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
# # data.loc[(data['Age'])>48&(data['Age']<=64),'Age_band']=3
# # data.loc[data['Age']>64,'Age_band']=4
# # data.head(2)


# # data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
# # sns.factorplot('Age_band','Survived',data=data, col='Pclass')


# # plt.show()


