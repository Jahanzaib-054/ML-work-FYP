import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import joblib


# Load dataset
data = pd.read_csv('ILPD.csv', delimiter=';')
data2 = pd.read_csv('hcc.csv', delimiter=';')

desired_order = ['TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'Target']

data2 = data2[desired_order]
# print(data.shape)
print('dataset ILPD : ',data.columns)
print('HCC dataset : ',data2.columns)

# # Define features and target variable
# X = data.drop(columns=['Target', 'A/G Ratio'])
# y = data['Target']

# test_x = data2.drop(columns=['Target'])
# test_y = data2['Target']
# print(X)
# print(y)
# print(test_x)

X = data.drop(columns=['Target', 'A/G Ratio'])
y = data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# Define AdaBoost classifier with DecisionTree base estimator
base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=2, min_samples_split=2)
ada_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=10, learning_rate=0.1)

# Train the model
ada_boost.fit(X_train, y_train)

# Make predictions
y_pred = ada_boost.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

joblib.dump(ada_boost, 'ada_boost_model.pkl')


# print("Best Parameters:", grid_search.best_params_)
# print("Accuracy: {:.2f}".format(accuracy))
# print("F1 Score: {:.4f}".format(f1))
# print("Accuracy:", accuracy)
# print("F1 Score:", f1)
