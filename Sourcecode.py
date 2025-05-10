from google.colab import files

import pandas as pd

df = pd.read_excel('dataset.csv.xlsx')  # Excel file
df.head()

df.info()
df.describe()
df.columns.tolist()

print("Missing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

import seaborn as sns
import matplotlib.pyplot as plt

# Histogram for a numeric column (replace with your actual column)
sns.histplot(df['flags'], kde=True)
plt.title('Math Score Distribution')
plt.show()

# Boxplot of multiple numeric columns (update with yours)
sns.boxplot(data=df[['flags', 'category', 'intent']]) # Remove extra spaces in column names
plt.title('Boxplot of Subject Scores')
plt.show()

X = df.drop('utterance', axis=1)
y = df['utterance']

for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes

X = pd.get_dummies(X, drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Import necessary libraries
from sklearn.preprocessing import LabelEncoder

# ... (Your existing code) ...

# 7. Convert Categorical Columns to Numerical
# ... (Your existing code) ...

# Convert 'utterance' to numerical labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Apply to your original 'y' before train-test split

# ... (Rest of your code with train-test split, model selection, etc.) ...

# Now use an appropriate classification model instead of RandomForestRegressor
from sklearn.linear_model import LogisticRegression #Example Model

model = LogisticRegression()

# Import necessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression #Example Model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ... (Your existing code for data loading, preprocessing, etc.) ...

# Convert 'utterance' to numerical labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Apply to your original 'y' before train-test split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train) # This line is crucial to train the model

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Example new student data (replace with your values and feature order)
new_input = [[16, 90, 85, 92]]  # Age, Math, Reading, Writing

# Create a DataFrame using the same column names as your original data
new_df = pd.DataFrame(new_input, columns=['flags', 'category', 'intent', 'utterance'])  # replace 'flags', 'category', 'intent', 'utterance' with the correct column names from X

# Apply any necessary data transformations (encoding, etc.) to make new_df similar to X
# This section needs the exact steps you used during preprocessing like categorical encoding
# Refer to the steps you used to create X_scaled

# Example of encoding categorical data (if needed)
for col in new_df.select_dtypes(include='object').columns:
    new_df[col] = new_df[col].astype('category').cat.codes

# Example of one-hot encoding (if needed)
new_df = pd.get_dummies(new_df, drop_first=True)

# Check if the number of features in new_df matches X_scaled
print(f"new_df shape: {new_df.shape}")
print(f"X_scaled shape: {X_scaled.shape}")

# Align columns if number of features don't match
missing_cols = set(X.columns) - set(new_df.columns)
for col in missing_cols:
    new_df[col] = 0  # Add missing columns with value 0
new_df = new_df[X.columns]  # Reorder columns

new_scaled = scaler.transform(new_df)  # Now scale the properly formatted data
model.predict(new_scaled)

# Example new student data (replace with your values and feature order)
new_input = [[16, 90, 85, 92]]  # Age, Math, Reading, Writing

# Create a DataFrame with the original column names
new_df = pd.DataFrame(new_input, columns=['flags', 'category', 'intent', 'utterance'])

# Apply the same preprocessing steps used for the training data
# ... (Your preprocessing code here, e.g., encoding, scaling) ...

# Example: Assuming you have the same preprocessing steps as before
for col in new_df.select_dtypes(include='object').columns:
    new_df[col] = new_df[col].astype('category').cat.codes

new_df = pd.get_dummies(new_df, drop_first=True)

# Align columns
missing_cols = set(X.columns) - set(new_df.columns)
for col in missing_cols:
    new_df[col] = 0  # Add missing columns with value 0
new_df = new_df[X.columns]  # Reorder columns


new_scaled = scaler.transform(new_df)  # Scale the data

predicted_grade = model.predict(new_scaled)
print("Predicted Final Grade:", round(predicted_grade[0], 2))

predicted_grade = model.predict(new_scaled)
print("Predicted Final Grade:", round(predicted_grade[0], 2))

!pip install gradio
import gradio as gr

def predict_grade(age, math, reading, writing):
    input_data = [[age, math, reading, writing]]
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    return round(prediction[0], 2)

interface = gr.Interface(
    fn=predict_grade,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Math Score"),
        gr.Number(label="Reading Score"),
        gr.Number(label="Writing Score"),
    ],
    outputs="number",
    title="🎓 Student Performance Predictor",
    description="Enter scores to predict the student's final grade."
)

interface.launch()
