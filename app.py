from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_and_split_data(df):
    # Fill missing values and encode data
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    df.replace({
        "Loan_Status": {'N': 0, 'Y': 1},
        "Gender": {'Male': 0, 'Female': 1},
        "Education": {'Not Graduate': 0, 'Graduate': 1},
        "Married": {'No': 0, 'Yes': 1},
        "Self_Employed": {'No': 0, 'Yes': 1}
    }, inplace=True)

    y = df["Loan_Status"]
    x = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
    x = pd.get_dummies(data=x, columns=["Property_Area", "Dependents"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    return x_train, x_test, y_train, y_test, list(x.columns)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Load and preprocess data
        df = pd.read_csv(file_path)
        x_train, x_test, y_train, y_test, features = preprocess_and_split_data(df)

        # Train and evaluate model
        model = RandomForestClassifier(random_state=0, max_depth=5, min_samples_split=0.01,
                                        max_features=0.8, max_samples=0.8)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        # Feature importance plot
        importances = pd.DataFrame(model.feature_importances_)
        importances['features'] = features
        importances.columns = ['importance', 'feature']
        importances.sort_values(by='importance', ascending=True, inplace=True)

        # Plotting Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(importances.feature, importances.importance, color='skyblue')
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")

        # Save feature importance plot to buffer
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        feature_importance_graph = base64.b64encode(img.getvalue()).decode()
        feature_importance_graph = f'data:image/png;base64,{feature_importance_graph}'
        plt.close()

        # Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Save confusion matrix plot to buffer
        cm_img = BytesIO()
        plt.savefig(cm_img, format='png')
        cm_img.seek(0)
        cm_graph = base64.b64encode(cm_img.getvalue()).decode()
        cm_graph = f'data:image/png;base64,{cm_graph}'
        plt.close()

        # Function to plot categorical graphs with color differentiation
        def plot_categorical_graph(column, df):
            plt.figure(figsize=(8, 6))
            sns.countplot(x=column, data=df, palette='Set2')  # Set2 palette for color differentiation
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_b64 = base64.b64encode(img.getvalue()).decode()
            img_b64 = f'data:image/png;base64,{img_b64}'
            plt.close()
            return img_b64

        # Create graphs for each feature
        gender_graph = plot_categorical_graph('Gender', df)
        married_graph = plot_categorical_graph('Married', df)
        dependents_graph = plot_categorical_graph('Dependents', df)
        self_employed_graph = plot_categorical_graph('Self_Employed', df)
        credit_history_graph = plot_categorical_graph('Credit_History', df)

        return render_template('result.html', accuracy=accuracy, conf_matrix=conf_matrix, report=report,
                               feature_importance_graph=feature_importance_graph, cm_graph=cm_graph,
                               gender_graph=gender_graph, married_graph=married_graph,
                               dependents_graph=dependents_graph, self_employed_graph=self_employed_graph,
                               credit_history_graph=credit_history_graph)

if __name__ == '__main__':
    app.run(debug=True)


https://github.com/0422msrinivas/financeWeb
git remote add origin https://github.com/0422msrinivas/FinanceWebApp.git
