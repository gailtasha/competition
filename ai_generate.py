import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data from CSV files
train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")

# Preprocess the data by converting target variable into numerical values (0 and 1)
le = LabelEncoder()
train_df["target"] = le.fit_transform(train_df["target"])
test_df["target"] = le.fit_transform(test_df["target"])

# Split the data into features (X) and target (y)
X_train, X_test, y_train, y_test = train_test_split(train_df.drop("target", axis=1), train_df["target"], test_size=0.2, random_state=42)

# Define a simple logistic regression model
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_test = model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Use the model to make predictions on the test data
test_results = []
for index, row in test_df.iterrows():
    prediction = model.predict([row])
    test_results.append(prediction)

# Print the results
print("Test Results:")
for i, result in enumerate(test_results):
    print(f"Sample {i+1}: {test_df.iloc[i]['target']} -> {result}")

