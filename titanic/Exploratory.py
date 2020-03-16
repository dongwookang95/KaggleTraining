import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython


plt.style.use('seaborn')

import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# what is this?
# It is called magic function. Which is calling line-oriented and cell oriented.
# The %matplotlib inline will make your plot outputs appear and be stored within the notebok.
'exec(%matplotlib inline)'

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.head()


# it shows count, mean , std, min, 25%, 50%, 75%, max. 
# So basically automatic calc for each feature. 
# print(df_train.describe())
# print(df_test.describe())

#####################
###Null data check###
#####################

# for col in df_train.columns:
#     msg = 'column: {:10}\t Percent of Nan value: {:.2f}%'.format(col,
#     100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
#     print(msg)

# print("###############")

# for col in df_test.columns:
#     msg = 'column: {:10}\t Percent of NaN value: {:.2f}%'.format(col,
#     100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
#     print(msg)

###Graph execution###

msno.matrix(df=df_train.iloc[:, :], figsize = (8,8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_train.iloc[:, :], figsize=(8,8), color=(0.8, 0.5, 0.2))


# plt.show()
###Check Target label###

f, ax = plt.subplots(1, 2, figsize=(18, 8))

#what are those
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1]
,autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')

# plt.show()

# df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).count()

# df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()

#above procedure can be simplified by crosstab of pandas.

pd.crosstab(df_train['Pclass'], df_train['Survived'],
 margins=True).style.background_gradient(cmap='summer_r')

df_train[['Pclass','Survived']].groupby(['Pclass'
], as_index=True).mean().sort_values(by='Survived', ascending = False).plot.bar()
# plt.show()

# y_position = 1.02
# f, ax = plt.subplots(1, 2, figsize =(18, 8))
# df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],
# ax=ax[0])
# ax[0].set_title('Number of Passengers By Pclass', y=y_position)
# ax[0].set_ylabel('Count')
# sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
# ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
# plt.show()
###Above was depending on Pclass
###Below from here is depending on sex

y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18,8))
###look up this line
df_train[['Sex','Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')


###Let's take into account both of stuffs

sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6,aspect=1.5)
sns.factorplot(x = 'Sex',y = 'Survived', col='Pclass', data=df_train, size=6,aspect=1.5)


### Let's look up Age feature
print('The oldest : {:.1f} Year'.format(df_train['Age'].max()))
print('The youngest : {:.1f} Year'.format(df_train['Age'].min()))
print('Average : {:.1f} Year'.format(df_train['Age'].mean()))

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
#ax=ax 이건 볼수록 모르겠다
###histogram according to the survival age.
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'SUrvived == 0'])


plt.figure(figsize = (8,6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])


plt.figure(figsize=(8,6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st class', '2nd class', '3rd Class'])

### We checked as class growth, 
cummulate_survival_ratio = []
for i in range(1,80):
    cummulate_survival_ratio.append(df_train[df_train['Age']<i]['Survived'].sum() /
    len(df_train[df_train['Age']<i]['Survived']))

### we see strong correlation between age and survival rate
plt.figure(figsize=(7,7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')


###figsize가 뭔지 확실히 확인하기
f, ax = plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age",hue="Survived", data=df_train, scale = 'count', split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Servived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age",hue="Survived",data=df_train,scale='count',
split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))


###Depends on Embarked
f, ax = plt.subplots(1, 1, figsize=(7,7))
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(
by='Survived', ascending=False).plot.bar(ax=ax)

f, ax=plt.subplots(2, 2, figsize=(20,15))
sns.countplot('Embarked', data=df_train, ax=ax[0,0])
ax[0,0].set_title('(1) No. Of passengers Boarded')
sns.countplot('Embarked', hue = 'Sex', data=df_train, ax=ax[0,1])
ax[0,1].set_title('(2) Male-Female Split for Embarked')
sns.countplot('Embarked', hue = 'Survived', data = df_train, ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs Survived')
sns.countplot('Embarked', hue = 'Pclass', data=df_train, ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)


###Depends on family size
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
### need to include himself
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

print("Maximum size of Family :", df_train['FamilySize'].max())
print("Minimum size of Family :", df_train['FamilySize'].min())

f,ax=plt.subplots(1, 3, figsize=(40,10))
sns.countplot('FamilySize', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(
    by='Survived',ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)



###Depends on Fare
###보시다시피, distribution이 매우 비대칭인 것을 알 수 있습니다.(high skewness). 만약 이대로 모델에 넣어준다면 자칫 모델이 잘못 학습할 수도 있습니다. 몇개 없는 outlier 에 대해서 너무 민감하게 반응한다면, 실제 예측 시에 좋지 못한 결과를 부를 수 있습니다.
#outlier의 영향을 줄이기 위해 Fare 에 log 를 취하겠습니다.
#여기서 우리는 pandas 의 유용한 기능을 사용할 겁니다. dataFrame 의 특정 columns 에 공통된 작업(함수)를 적용하고 싶으면 아래의 map, 또는 apply 를 사용하면 매우 손쉽게 적용할 수 있습니다.
#우리가 지금 원하는 것은 Fare columns 의 데이터 모두를 log 값 취하는 것인데, 파이썬의 간단한 lambda 함수를 이용해 간단한 로그를 적용하는 함수를 map 에 인수로 넣어주면, Fare columns 데이터에 그대로 적용이 됩니다. 매우 유용한 기능이니 꼭 숙지하세요!
fig, ax = plt.subplots(1,1,figsize=(8,8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g= g.legend(loc='best')

df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() # testset 에 있는 nan value 를 평균값으로 치환합니다.

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')


### Cabin
### this feature has 80% of NaN, meaning that useless information 
### So we are going to exclude them.
