# Set up code checking


# Set up filepaths
import os
if not os.path.exists("/content/train.csv"):
    os.symlink("/content/train.csv")
    os.symlink("/content/train.csv") 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
iowa_file_path = '/content/train.csv'
home_data = pd.read_csv(iowa_file_path)
home_data.dropna()
y = home_data.SalePrice
# Create X (After completing the exercise, you can return to modify this line!)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select columns corresponding to features, and preview the data
X = home_data[features]
X.dropna()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X,y)
rf_model_val_pred_on_full_data= rf_model_on_full_data.predict(val_X)
rf_val_mae_full = mean_absolute_error(rf_model_val_pred_on_full_data, val_y)
# fit rf_model_on_full_data on all data from the training data
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae_full))
# path to file you will use for predictions
test_data_path = '/content/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)


# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features

test_X= test_data[features]
# make predictions which we will submit. 
test_preds =  rf_model_on_full_data.predict(test_X)
print(test_preds)
