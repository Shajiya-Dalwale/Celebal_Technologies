import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'F:\\Celebal\\Assignment_3\\Titanic-Dataset.csv'  
titanic = pd.read_csv(file_path)

print(titanic.head())

died = titanic[titanic['Survived'] == 0]

plt.figure(figsize=(8, 6))
sns.countplot(data=died, x='Sex', palette='coolwarm')

plt.title('Number of Male and Female Passengers Who Died on the Titanic')
plt.xlabel('Sex')
plt.ylabel('Count')

plt.show()
