from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
df = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global df
    chart_url = None
    cm_url = None
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        
        # Remove columns with minimal diversity
        df = df.drop(['Bwd PSH Flags',
                      'Fwd URG Flags',
                      'Bwd URG Flags',
                      'CWE Flag Count',
                      'Fwd Avg Bytes/Bulk',
                      'Fwd Avg Packets/Bulk',
                      'Fwd Avg Bulk Rate',
                      'Bwd Avg Bytes/Bulk',
                      'Bwd Avg Packets/Bulk',
                      'Bwd Avg Bulk Rate'], axis=1, errors='ignore')

        # Data Cleanup - Remove Infinity, Not Available and Deuplicate records
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        
        # Check for infinite values in each numeric column only
        numeric_cols = df.select_dtypes(include=[np.number]) # This selects only numeric columns
        # Create a mask where any row contains at least one infinite value
        rows_with_inf = numeric_cols.apply(lambda x: np.isinf(x)| np.isneginf(x)).any(axis=1)
        #rows with infinite values
        rows_with_inf = df[rows_with_inf]
        # Drop rows with missing values that have 'NetBIOS' label
        df = df.drop(rows_with_inf.index)
        # Reset index if needed
        df.reset_index(drop=True, inplace=True)        

        columns = df.columns.tolist()
        return render_template('index.html', columns=columns, chart_url=chart_url, cm_url=cm_url)
    return render_template('index.html', columns=None, chart_url=chart_url, cm_url=cm_url)

@app.route('/plot', methods=['POST'])
def plot():
    global df
    # target = request.form['target']
    target = 'Label'
    model_name = request.form['model']

    if df is None:
        return "No data uploaded."

    if target not in df.columns:
        return "Invalid target column."

    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    # Drop infinite and NaN
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[:len(X)]  # Ensure same length
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_mapping = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': CalibratedClassifierCV(LinearSVC())
    }

    model = model_mapping.get(model_name)
    if model is None:
        return "Unsupported model."

    os.makedirs("static", exist_ok=True)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = "static/roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(cm)
    disp = sns.heatmap(cm,annot = True,cmap='Blues')
    disp.plot()
    cm_path = 'static/confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()

    metrics = {
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1Score': f"{f1:.4f}",
        'AUC': f"{auc_score:.4f}" if auc_score is not None else "N/A"
    }

    return render_template('index.html', columns=df.columns.tolist(), chart_url=f'/{roc_path}', metrics=metrics, cm_url=f'/{cm_path}')

if __name__ == '__main__':
    app.run(debug=True)
