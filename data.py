import pandas as pd
import sklearn 
import joblib
df = pd.read_csv("customers-100.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

X = df.drop("Purchased", axis = 1)

y = df["Purchased"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

numeric_features = [
    "Age",
    "Income",
    "Experience"
]

categorical_features = [
    "Education"
]

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

Pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
        ("model", LogisticRegression())
    ]
)


Pipeline([ 
    ("scaler", StandardScaler()),
    ("model",SVC())
])

Pipeline.fit(X_train, y_train)
y_pred = Pipeline.predict(X_test)

accuracy =  accuracy_score(
    y_test,
    y_pred
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

scores = cross_val_score(
    Pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print(scores)
print(scores.mean())

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)

params = {
    "model__n_estimators":[100,200],
    "model__max_depth":[5,10,None]
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
print(gird.best_params_)

final_model = grid.best_estimator_
y_pred = final_model.predict(X_test)

print(
    accuracy_score(
        y_test,
        y_pred
    )
)

joblib.dump(
    final_model,
    "model.pkl"
)

model = joblib.load(
    "model.pkl"
)

prediction = model.predict(new_data)