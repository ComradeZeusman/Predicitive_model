import pandas as pd

df = pd.read_csv('dataset.csv')


#split the data set into x and y
x = df.drop('Target', axis=1)
y = df['Target']

#split the data set into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

model.fit(x_train, y_train_encoded)

#predict the test set
y_model_train_pred = model.predict(x_train)
y_model_test_pred = model.predict(x_test)

#evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
model_train_mse = mean_squared_error(y_train_encoded, y_model_train_pred)
model_train_r2 = r2_score(y_train_encoded, y_model_train_pred)

model_test_mse = mean_squared_error(y_test_encoded, y_model_test_pred)
model_test_r2 = r2_score(y_test_encoded, y_model_test_pred)

print('Train MSE: ', model_train_mse)
print('Train R2: ', model_train_r2)
print('Test MSE: ', model_test_mse)
print('Test R2: ', model_test_r2)

