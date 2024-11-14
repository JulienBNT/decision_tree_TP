# J'ai perdu la connexion vers 11h hier, j'ai du faire l'arbre
# avec les données du iris.csv
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv("iris.csv")
print(iris_data['Species'].unique())

X = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = iris_data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_split=2, min_samples_leaf=1, random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(16, 10))
plot_tree(clf, feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
          class_names=clf.classes_, filled=True)
plt.title("Arbre de Décision des Iris")
plt.show()

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
species_colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'green', 'Iris-virginica': 'red'}
for species, color in species_colors.items():
    subset = iris_data[iris_data['Species'] == species]
    plt.scatter(subset['SepalLengthCm'], subset['SepalWidthCm'], label=species, color=color, alpha=0.7)

plt.xlabel("Longueur des Sépales (cm)")
plt.ylabel("Largeur des Sépales (cm)")
plt.title("Longueur vs Largeur des Sépales par Espèce")
plt.legend(title='Espèce')
plt.grid(True)
plt.show()
