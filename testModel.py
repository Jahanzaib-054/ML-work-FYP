import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

model = joblib.load('ada_boost_model.pkl')

data = pd.read_csv('hcc.csv', delimiter=';')

print('HCC dataset : ',data.columns)

desired_order = ['TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'Target']

data = data[desired_order]

test_x = data.drop(columns=['Target'])
test_y = data['Target']

# print(test_x)
# print(test_y)
# print(test_x.iloc[4:5,:])
# print(test_y.iloc[4:5,])

# prediction = model.predict(test_x.iloc[4:5,:])

y_pred = model.predict(test_x)
accuracy = accuracy_score(test_y, y_pred)
f1 = f1_score(test_y, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)

# print('prediction : ',prediction)