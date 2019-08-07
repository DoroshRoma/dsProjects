import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Read the data
X = pd.read_csv('./data/train.csv', index_col='Id')
X_test_full = pd.read_csv('./data/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y,
                                                                random_state=0)

# "Cardinality" numbers of unique values in a column
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10
                        and X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype
                in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(imputer.transform(X_valid))

final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

model = RandomForestRegressor(max_leaf_nodes=325, n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

preds_valid = model.predict(final_X_valid)
print("MAE:")
print(mean_absolute_error(y_valid, preds_valid))