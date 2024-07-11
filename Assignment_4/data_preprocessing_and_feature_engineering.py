import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_path = r'F:\\Celebal\Assignment_3\\titanic-dataset.csv'
titanic = pd.read_csv(file_path)

print(titanic.head())

# Step 1: Handle Missing Values
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

titanic.drop(columns=['Cabin'], inplace=True)

# Step 2: Encode Categorical Variables
titanic['Sex'] = LabelEncoder().fit_transform(titanic['Sex'])

titanic = pd.get_dummies(titanic, columns=['Embarked'], drop_first=True)

# Step 3: Feature Engineering
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

titanic['IsAlone'] = 1
titanic['IsAlone'].loc[titanic['FamilySize'] > 1] = 0

titanic.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Step 4: Standardize/Normalize Data
scaler = StandardScaler()
titanic[['Age', 'Fare']] = scaler.fit_transform(titanic[['Age', 'Fare']])

print(titanic.head())

titanic.to_csv(r'F:\\Celebal\Assignment_3\\titanic_preprocessed.csv', index=False)

died = titanic[titanic['Survived'] == 0]

plt.figure(figsize=(8, 6))
sns.countplot(data=died, x='Sex', palette='coolwarm')

plt.title('Number of Male and Female Passengers Who Died on the Titanic')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
