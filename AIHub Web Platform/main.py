import re
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, log_loss, zero_one_loss


app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///projects.db'
app.app_context()
db = SQLAlchemy(app)


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    file_data = db.Column(db.LargeBinary)


@app.route("/")
def home():
    return render_template("home.html")

## Dashboard

@app.route("/dashboard/")
def dashboard():
    total_projects = Project.query.count()
    return render_template("dashboard.html", total_projects=total_projects)


## Visulization

def create_visualizations(data):
    # Perform data visualization based on the type of data present in the table
    # Customize this function based on your specific visualization requirements
    try:
        # Numerical columns - create bar charts
        numerical_cols = data.select_dtypes(include=['int', 'float']).columns
        for col in numerical_cols:
            data[col].value_counts().plot(kind='bar')
            plt.title(f'Bar chart for {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

        # Categorical columns - create bar charts
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col].value_counts().plot(kind='bar')
            plt.title(f'Bar chart for {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

        # Date columns - create line charts
        date_cols = data.select_dtypes(include=['datetime']).columns
        for col in date_cols:
            data[col].value_counts().plot(kind='line')
            plt.title(f'Line chart for {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

        # Location columns - create scatter plots
        location_cols = [col for col in data.columns if 'latitude' in col.lower() or 'longitude' in col.lower() or 'city' in col.lower() or 'country' in col.lower()]
        for col in location_cols:
            fig, ax = plt.subplots()
            ax.scatter(data[col + '_longitude'], data[col + '_latitude'])
            ax.set_title(f'Scatter plot for {col}')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.show()

    except Exception as e:
        error_message = str(e)
        return error_message
    
def suggest_visualization(data):
    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['int', 'float']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    date_cols = data.select_dtypes(include=['datetime']).columns

    # Analyze numerical columns
    numerical_cols_best = []

    for col in numerical_cols:
        numerical_cols_best.append(col)

    # Analyze categorical columns
    categorical_cols_best = []

    for col in categorical_cols:
        categorical_cols_best.append(col)

    # Analyze date columns
    date_cols_best = []

    for col in date_cols:
        date_cols_best.append(col)

    # Return the suggested visualization types and best columns
    return {
        'numerical': {
            'columns': numerical_cols_best
        },
        'categorical': {
            'columns': categorical_cols_best
        },
        'date': {
            'columns': date_cols_best
        }
    }

@app.route("/dashboard/visualization/", methods=['GET', 'POST'])
def visualizationspage():
    total_projects = Project.query.all()
    if request.method == 'POST':
        try:
            data = pd.read_csv('countries.csv')  # Replace 'data.csv' with your data source
            suggested_visualizations = suggest_visualization(data)
            return render_template('visulizations.html',total_projects= total_projects, suggested_visualizations=suggested_visualizations)
        except Exception as e:
            error_message = f'Error loading data: {e}'
            return render_template('error.html', error_message=error_message)
        
    return render_template("visulizations.html", total_projects = total_projects)


## Datasets

@app.route("/dashboard/datasets/upload/", methods=['GET', 'POST'])
def uploaddatasets():
    if request.method == 'POST':
        name = request.form['title']
        description = request.form['description']
        file = request.files['file']
        project = Project(name=name, description=description, file_data=file.read())
        db.session.add(project)
        db.session.commit()
        return redirect(url_for('detaildatasets', project_id=project.id))
    return render_template("uploaddataset.html")

@app.route("/dashboard/datasets/<int:project_id>/", methods=['GET', 'POST'])
def detaildatasets(project_id):
    project = Project.query.get_or_404(project_id)

    if request.method == 'POST':
        # Perform table operations based on user selection
        selected_functions = request.form.getlist('functions')
        selected_columns = request.form.getlist('columns')
        table_data = decode_table_data(project.file_data)
        table_data = apply_table_operations(table_data, selected_functions, selected_columns)
    else:
        table_data = decode_table_data(project.file_data)

    headers = table_data[0]
    data_rows = table_data[1:]

    # Search functionality
    search_term = request.args.get('search')
    if search_term:
        data_rows = search_data(data_rows, search_term)

    # Sort functionality
    sort_by = request.args.get('sort')
    if sort_by:
        data_rows = sort_data(data_rows, sort_by)

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of rows to display per page
    total_rows = len(data_rows)
    total_pages = ceil(total_rows / per_page)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_data_rows = data_rows[start_idx:end_idx]

    return render_template('datasetdetail.html', project=project, headers=headers, data_rows=paginated_data_rows, total_pages=total_pages, current_page=page)


def decode_table_data(file_data):
    file_data_str = file_data.decode('utf-8')
    rows = file_data_str.split('\n')
    table_data = [row.split(',') for row in rows if row]
    return table_data


def apply_table_operations(table_data, selected_functions, selected_columns):
    # Convert column names to column indices
    column_indices = [table_data[0].index(column) for column in selected_columns]

    if 'drop_na' in selected_functions:
        table_data = drop_na(table_data, column_indices)
    if 'drop_duplicates' in selected_functions:
        table_data = drop_duplicates(table_data)
    if 'drop_null' in selected_functions:
        table_data = drop_null_values(table_data, column_indices)
    if 'fill_null' in selected_functions:
        table_data = fill_null_values(table_data, column_indices)
    if 'standardize' in selected_functions:
        table_data = standardize_values(table_data, column_indices)
    if 'normalize' in selected_functions:
        table_data = normalize_values(table_data, column_indices)
    return table_data


def drop_na(table_data, selected_columns):
    for i in range(len(table_data)):
        row = table_data[i]
        if all(row[j] is None or row[j] == '' for j in selected_columns):
            table_data.pop(i)
    return table_data


def drop_duplicates(table_data):
    return [table_data[0]] + [row for i, row in enumerate(table_data[1:], start=1) if row != table_data[i - 1]]


def drop_null_values(table_data, selected_columns):
    return [row for row in table_data if not any(row[column] is None or row[column] == '' for column in selected_columns)]


def fill_null_values(table_data, selected_columns):
    for column in selected_columns:
        values = [row[column] for row in table_data if row[column] is not None and row[column] != '']
        mean = sum(values) / len(values)
        for row in table_data:
            if row[column] is None or row[column] == '':
                row[column] = mean
    return table_data


def standardize_values(table_data, selected_columns):
    for column in selected_columns:
        values = [row[column] for row in table_data if row[column] is not None and row[column] != '']
        mean = sum(values) / len(values)
        std_dev = (sum((val - mean) ** 2 for val in values) / len(values)) ** 0.5
        for row in table_data:
            if row[column] is not None and row[column] != '':
                row[column] = (row[column] - mean) / std_dev
    return table_data


def normalize_values(table_data, selected_columns):
    for column in selected_columns:
        values = [row[column] for row in table_data if row[column] is not None and row[column] != '']
        min_val = min(values)
        max_val = max(values)
        for row in table_data:
            if row[column] is not None and row[column] != '':
                row[column] = (row[column] - min_val) / (max_val - min_val)
    return table_data


def search_data(data_rows, search_term):
    if not search_term:
        return data_rows
    search_results = []
    for row in data_rows:
        for cell in row:
            if search_term.lower() in str(cell).lower():
                search_results.append(row)
                break
    return search_results


def sort_data(data_rows, sort_by):
    if not sort_by:
        return data_rows

    clean_sort_by = re.sub(r'\W+', '', sort_by)  # Remove all non-alphanumeric characters
    header_index = next((index for index, header in enumerate(data_rows[0]) if re.sub(r'\W+', '', header) == clean_sort_by), None)
    if header_index is None:
        raise ValueError(f"Column '{sort_by}' does not exist in the data.")
    sorted_rows = sorted(data_rows[1:], key=lambda x: x[header_index])
    return [data_rows[0]] + sorted_rows


@app.route("/dashboard/datasets/public/")
def publicdatasets():
    total_projects = Project.query.all()
    return render_template("publicdataset.html", total_projects=total_projects)

@app.route("/dashboard/datasets/private/")
def privatedatasets():
    total_projects = Project.query.all()
    return render_template("yourdataset.html", total_projects=total_projects)

@app.route("/dashboard/datasets/edit/")
def editdatasets():
    return render_template("datasetedit.html")


## Machine Learning

@app.route('/dashboard/ml/supervised/', methods=['GET', 'POST'])
def supervised():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            data = pd.read_csv(file)
            columns = data.columns.tolist()
            print("Step 1 Complete")
            return render_template('supervised.html', step=2, columns=columns)
        
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        columns = pd.read_csv(file_path).columns.tolist()
        return render_template('select_columns.html', filename=filename, columns=columns)
    return render_template('upload.html')

@app.route('/select_columns', methods=['POST'])
def select_columns():
    if request.method == 'POST':
        filename = request.form['filename']
        selected_columns = request.form.getlist('selected_columns')
        target_column = request.form['target_column']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        columns = pd.read_csv(file_path, usecols=selected_columns + [target_column]).columns.tolist()
        session['columns'] = columns
        print("Selected Columns = ", selected_columns)
        print("Target Column = ", target_column)
        print("Columns = ", columns)
        print("Slect Columns ends here")
        return render_template('split_data.html', filename=filename, columns=columns, target_column=target_column)

@app.route('/split_data', methods=['POST'])
def split_data():
    if request.method == 'POST':
        filename = request.form['filename']
        selected_columns = session.get('columns', [])
        # selected_columns = request.form.getlist('selected_columns')
        target_column = request.form['target_column']
        test_size = float(request.form['test_size'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("Selected Columns = ", selected_columns)
        print("Target Columns = ", target_column)
        columns = pd.read_csv(file_path, usecols=selected_columns + [target_column]).columns.tolist()
        print("All Columns = ", columns)
        print("Split Data Ends here")
        data = pd.read_csv(file_path)
        X = data[selected_columns]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return render_template('select_model.html', filename=filename, columns=columns, X_train=X_train.values.tolist(), X_test=X_test.values.tolist(), y_train=y_train.values.tolist(), y_test=y_test.values.tolist())

@app.route('/select_model', methods=['POST'])
def select_model():
    if request.method == 'POST':  
        filename = request.form['filename']
        model_name = request.form['model_name']
        columns = request.form.getlist('columns')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        X_train = pd.DataFrame(eval(request.form['X_train']))
        X_test = pd.DataFrame(eval(request.form['X_test']))
        y_train = pd.DataFrame(eval(request.form['y_train']))
        y_test = pd.DataFrame(eval(request.form['y_test']))
        print("Model selected = ", model_name)
        if model_name == 'logistic_regression':
            model = LogisticRegression()
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier()
        elif model_name == 'random_forest':
            model = RandomForestClassifier()
        elif model_name == 'kneighbors_classifier':
            model = KNeighborsClassifier()
        elif model_name == 'svc':
            model = SVC()
        elif model_name == 'gaussian':
            model = GaussianNB()
        else:
            return "Invalid Model Selected"
            # model = LogisticRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_score_ = f1_score(y_test, y_pred, average="binary")
        hamming_loss_ = hamming_loss(y_test, y_pred)
        log_loss_ = log_loss(y_test, y_pred)
        zero_one_loss_ = zero_one_loss(y_test, y_pred)

        return render_template('results.html', model_name=model_name, columns=columns, accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score_, hamming_loss=hamming_loss_, log_loss=log_loss_, zero_one_loss=zero_one_loss_)


@app.route("/dashboard/ml/unsupervised/")
def unsupervised():
    return render_template("unsupervised.html")


@app.route("/dashboard/code/", methods=['POST'])
def code():
    if request.method == 'POST':
        code = request.form['code']
        try:
            # Create a dictionary to store the captured output and error messages
            output = {}
            
            # Redirect the standard output and error to our dictionary
            exec(code, {}, output)
            
            # Extract the captured output and error
            stdout = output.get('stdout', '')
            stderr = output.get('stderr', '')
            return render_template('code.html', stdout=stdout, stderr=stderr)
        except Exception as e:
            return render_template('code.html', stderr=str(e))
    return render_template('code.html')
    


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    with app.app_context():
        db.create_all()
        app.run(debug=True)
