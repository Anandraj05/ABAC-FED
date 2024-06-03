import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel


try:
    df = pd.read_csv('../merged_data.csv', sep=',')
except FileNotFoundError:
    print("Error: File 'merged_data.csv' not found")
    exit()


imputer = SimpleImputer(strategy='most_frequent')
df['device_type'] = imputer.fit_transform(df[['device_type']]).ravel()
df['attempt_time'] = pd.to_datetime(df['attempt_time'], format='%I:%M:%S %p', errors='coerce')
df['logout_time'] = pd.to_datetime(df['logout_time'], format='%I:%M:%S %p', errors='coerce')


le = LabelEncoder()
df['country'] = le.fit_transform(df['country'])
df['is_active'] = df['is_active'].map({'TRUE': 1, 'FALSE': 0})


df['time_of_access'] = pd.to_datetime(df['time_of_access'], format='%I:%M:%S %p').dt.hour
df['attempt_time'] = df['attempt_time'].dt.hour
df['last_login'] = pd.to_datetime(df['last_login'], format='%I:%M:%S %p').dt.hour
df['logout_time'] = df['logout_time'].dt.hour

df['session_duration'] = (df['logout_time'] - df['last_login']).clip(lower=0)

df['login_attempts'] = df.groupby('email')['attempt_time'].transform('count')
df['avg_session_duration'] = df.groupby('email')['session_duration'].transform('mean')
df['min_session_duration'] = df.groupby('email')['session_duration'].transform('min')
df['max_session_duration'] = df.groupby('email')['session_duration'].transform('max')
df['std_session_duration'] = df.groupby('email')['session_duration'].transform('std')
df['login_failure_rate'] = (df['login_attempts'] - df.groupby('email')['is_active'].transform('sum')) / df['login_attempts']

scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df[['login_attempts', 'avg_session_duration', 'min_session_duration', 'max_session_duration', 'std_session_duration', 'login_failure_rate']])
df[['login_attempts', 'avg_session_duration', 'min_session_duration', 'max_session_duration', 'std_session_duration', 'login_failure_rate']] = normalized_features


weights = {
    'login_attempts': 0.2,
    'avg_session_duration': 0.3,
    'min_session_duration': 0.1,
    'max_session_duration': 0.1,
    'std_session_duration': 0.1,
    'login_failure_rate': 0.2
}

df['behavior_score'] = (
    weights['login_attempts'] * df['login_attempts'] +
    weights['avg_session_duration'] * df['avg_session_duration'] +
    weights['min_session_duration'] * df['min_session_duration'] +
    weights['max_session_duration'] * df['max_session_duration'] +
    weights['std_session_duration'] * df['std_session_duration'] +
    weights['login_failure_rate'] * df['login_failure_rate']
)

df['behavior_score'] = df['behavior_score'] * 100


df = df.dropna(subset=['behavior_score'])

X = df.drop(['email', 'date_joined', 'behavior_score'], axis=1)
y = df['behavior_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = ['email_verified', 'time_of_access', 'attempt_time', 'last_login', 'logout_time', 'session_duration', 'login_attempts', 'avg_session_duration', 'min_session_duration', 'max_session_duration', 'std_session_duration', 'login_failure_rate']
categorical_features = ['country', 'device_type']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

rf_model = RandomForestRegressor(random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])


cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Cross-validation scores:", -cv_scores)
print("Mean cross-validation score:", -cv_scores.mean())


pipeline.fit(X_train, y_train)

selector = SelectFromModel(estimator=rf_model, prefit=True)
X_train_selected = selector.transform(preprocessor.transform(X_train))
X_test_selected = selector.transform(preprocessor.transform(X_test))


rf_model_selected = RandomForestRegressor(random_state=42)
rf_model_selected.fit(X_train_selected, y_train)


y_pred = rf_model_selected.predict(X_test_selected)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")


importances = rf_model_selected.feature_importances_
feature_names = preprocessor.get_feature_names_out()
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [feature_names[i] for i in selected_feature_indices]
importance_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("Feature Importances:")
print(importance_df)


df['predicted_behavior_score'] = rf_model_selected.predict(selector.transform(preprocessor.transform(df.drop(['email', 'date_joined', 'behavior_score'], axis=1))))

df['predicted_behavior_score'] = df['predicted_behavior_score'].clip(lower=0, upper=100)


df['absolute_behavior_score'] = df['predicted_behavior_score'].round().astype(int)


df.to_csv('dataset_with_absolute_behavior_scores.csv', index=False)