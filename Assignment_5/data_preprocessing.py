import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Data Cleaning
file_path = r'F:\\Celebal\Assignment_5\\titanic-dataset.csv'
titanic = pd.read_csv(file_path)

print(titanic.head())

# Step 2: Data Cleaning
titanic.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Step 3: Handle Missing Values
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)

titanic.drop(columns=['Cabin'], inplace=True)

# Step 4: Transformation
titanic['AgeGroup'] = pd.cut(titanic['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])

# Step 5: Normalize numerical features
scaler = StandardScaler()
titanic[['Age', 'Fare']] = scaler.fit_transform(titanic[['Age', 'Fare']])

# Step 6: Encode Categorical Variables
titanic['Sex'] = LabelEncoder().fit_transform(titanic['Sex'])

titanic = pd.get_dummies(titanic, columns=['Embarked', 'AgeGroup'], drop_first=True)

# Step 7: Feature Engineering
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

titanic['IsAlone'] = 1
titanic['IsAlone'].loc[titanic['FamilySize'] > 1] = 0

titanic.drop(columns=['SibSp', 'Parch'], inplace=True)

print(titanic.head())

X = titanic.drop(columns=['Survived'])
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv(r'F:\\Celebal\Assignment_5\\titanic_preprocessed_train.csv', index=False)
X_test.to_csv(r'F:\\Celebal\Assignment_5\\titanic_preprocessed_test.csv', index=False)
y_train.to_csv(r'F:\\Celebal\Assignment_5\\titanic_preprocessed_y_train.csv', index=False)
y_test.to_csv(r'F:\\Celebal\Assignment_5\\titanic_preprocessed_y_test.csv', index=False)

died = titanic[titanic['Survived'] == 0]

plt.figure(figsize=(8, 6))
sns.countplot(data=died, x='Sex', palette='coolwarm')

plt.title('Number of Male and Female Passengers Who Died on the Titanic')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
