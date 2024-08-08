# Finding-best-Diamond
In [7]:
#Load required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
In [15]:
# Load the dataset
dataset = pd.read_csv('WheatData.csv')
In [16]:
dataset.head()
Out[16]:
A	P	C	LK	WK	A_Coef	LKG	target
0	15.26	14.84	0.8710	5.763	3.312	2.221	5.220	0
1	14.88	14.57	0.8811	5.554	3.333	1.018	4.956	0
2	14.29	14.09	0.9050	5.291	3.337	2.699	4.825	0
3	13.84	13.94	0.8955	5.324	3.379	2.259	4.805	0
4	16.14	14.99	0.9034	5.658	3.562	1.355	5.175	0
In [17]:
#Key Statistics 
dataset.describe()
Out[17]:
A	P	C	LK	WK	A_Coef	LKG	target
count	210.000000	210.000000	210.000000	210.000000	210.000000	210.000000	210.000000	210.000000
mean	14.847524	14.559286	0.870999	5.628533	3.258605	3.700201	5.408071	1.000000
std	2.909699	1.305959	0.023629	0.443063	0.377714	1.503557	0.491480	0.818448
min	10.590000	12.410000	0.808100	4.899000	2.630000	0.765100	4.519000	0.000000
25%	12.270000	13.450000	0.856900	5.262250	2.944000	2.561500	5.045000	0.000000
50%	14.355000	14.320000	0.873450	5.523500	3.237000	3.599000	5.223000	1.000000
75%	17.305000	15.715000	0.887775	5.979750	3.561750	4.768750	5.877000	2.000000
max	21.180000	17.250000	0.918300	6.675000	4.033000	8.456000	6.550000	2.000000
In [21]:
# Import necessary libraries for classifiers and pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define feature matrix x and target vector y
x = dataset.drop('target', axis=1).to_numpy()
y = dataset['target'].to_numpy()

# Split the data into training and testing sets (80% train, 20% test)
# Use stratification to ensure each class is represented proportionally in the train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=100)

# Standardize the feature data (mean = 0, variance = 1)
# Fit the scaler on the training data and apply it to both the training and testing data
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)

# Initialize classifiers
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
In [22]:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Create a list to hold the pipelines
pipeline = []

# Pipeline for Random Forest Classifier with standard scaling
pipe_rdf = Pipeline([
    ('scl', StandardScaler()),  # Feature scaling
    ('clf', RandomForestClassifier(random_state=100))  # Random Forest Classifier
])
pipeline.insert(0, pipe_rdf)

# Pipeline for Decision Tree Classifier with standard scaling
pipe_dt = Pipeline([
    ('scl', StandardScaler()),  # Feature scaling
    ('clf', DecisionTreeClassifier(random_state=100))  # Decision Tree Classifier
])
pipeline.insert(1, pipe_dt)

# Set grid search parameters for hyperparameter tuning
modelpara = []

# Hyperparameters for Random Forest Classifier
param_gridrdf = {
    'clf__criterion': ['gini', 'entropy'],  # Criterion for splitting
    'clf__n_estimators': [100, 150, 200],   # Number of trees in the forest
    'clf__bootstrap': [True, False]         # Whether bootstrap samples are used
}
modelpara.insert(0, param_gridrdf)

# Hyperparameters for Decision Tree Classifier
max_depth = range(1, 100)  # Range of maximum depths for the tree
param_griddt = {
    'clf__criterion': ['gini', 'entropy'],  # Criterion for splitting
    'clf__max_depth': max_depth             # Maximum depth of the tree
}
modelpara.insert(1, param_griddt)
In [23]:
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Function to plot learning curves
def plot_learning_curves(model):
    # Compute learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,             # The model/estimator to use
        X=x_train,                   # Training feature data
        y=y_train,                   # Training target data
        train_sizes=np.linspace(0.1, 1.0, 10),  # Sizes of the training set
        cv=10,                       # Number of cross-validation folds
        scoring='recall_weighted',   # Scoring metric
        random_state=100             # Random state for reproducibility
    )
    
    # Calculate the mean and standard deviation of training scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Calculate the mean and standard deviation of validation scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curves
    plt.plot(train_sizes, train_mean, color='blue', marker='o', 
             markersize=5, label='Training recall')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                     alpha=0.15, color='blue')  # Shade the area representing the standard deviation

    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='Validation recall')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color='green')  # Shade the area representing the standard deviation
    
    # Customize plot
    plt.grid(True)
    plt.xlabel('Number of training samples')  
    plt.ylabel('Recall')                      
    plt.legend(loc='best')                    
    plt.ylim([0.5, 1.01])                     
    plt.show()                                
In [24]:
# Plot Learning Curve for Decision Tree
print('Decision Tree - Learning Curve')
plot_learning_curves(pipe_dt)

# Plot Learning Curve for Random Forest
print('\nRandom Forest - Learning Curve')
plot_learning_curves(pipe_rdf)
Decision Tree - Learning Curve
No description has been provided for this image
Random Forest - Learning Curve
No description has been provided for this image
In [25]:
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import seaborn as sns

# List of models for evaluation
models = []
models.append(('Decision Tree', pipe_dt))  # Decision Tree with pipeline
models.append(('Random Forest', pipe_rdf))  # Random Forest with pipeline

# Initialize lists to store evaluation results and model names
results = []
names = []
scoring = 'recall_weighted'  # Scoring metric for evaluation

# Model evaluation using repeated k-fold cross-validation
print('Model Evaluation - Recall')
for name, model in models:
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)  # Define repeated k-fold cross-validation
    cv_results = cross_val_score(model, x_train, y_train, cv=rkf, scoring=scoring)  # Compute cross-validated scores
    results.append(cv_results)  # Store results
    names.append(name)  # Store model name
    print('{} {:.2f} +/- {:.2f}'.format(name, cv_results.mean(), cv_results.std()))  # Print model performance

print('\n')

# Boxplot visualization of model performance
fig = plt.figure(figsize=(10, 5))  # Create a figure with specific size
fig.suptitle('Boxplot View')  # Set the title for the figure
ax = fig.add_subplot(111)  # Add subplot to the figure
sns.boxplot(data=results)  # Create a boxplot for the cross-validation results
ax.set_xticklabels(names)  # Set x-axis labels to model names
plt.ylabel('Recall')  # Set y-axis label
plt.xlabel('Model')  # Set x-axis label
plt.show()  # Display the plot
Model Evaluation - Recall
Decision Tree 0.94 +/- 0.05
Random Forest 0.93 +/- 0.05


No description has been provided for this image
In [26]:
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Function for performing Grid Search with cross-validation
def Gridsearch_cv(model, params):
    # Define cross-validation strategy
    cv2 = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)
    
    # Initialize GridSearchCV with the model, parameter grid, and cross-validation strategy
    gs_clf = GridSearchCV(model, params, cv=cv2, scoring='recall_weighted')
    
    # Fit GridSearchCV to the training data
    gs_clf = gs_clf.fit(x_train, y_train)
    
    # Retrieve the best model found during the grid search
    model = gs_clf.best_estimator_
    
    # Use the best model to make predictions on the test data
    y_pred = model.predict(x_test)
    
    # Identify the best parameters for the model
    bestpara = str(gs_clf.best_params_)
    
    # Output the optimized model details
    print('\nOptimized Model')
    print('\nModel Name:', str(model.named_steps['clf']))
    print('\n')
    
    # Output feature importances for the best model
    print('Feature Importances')
    for name, score in zip(list(dataset), gs_clf.best_estimator_.named_steps['clf'].feature_importances_):
        print(name, round(score, 2))
    
    # Output validation statistics
    target_names = ['Kama', 'Rosa', 'Canadian']  # Adjust these as necessary for your target classes
    print('\nBest Parameters:', bestpara)
    print('\n', confusion_matrix(y_test, y_pred))  # Confusion matrix
    print('\n', classification_report(y_test, y_pred, target_names=target_names))  # Classification report
In [27]:
# Run Grid Search for each model and parameter grid
for pipe, params in zip(pipeline, modelpara):
    Gridsearch_cv(pipe, params)
Optimized Model

Model Name: RandomForestClassifier(bootstrap=False, n_estimators=200, random_state=100)


Feature Importances
A 0.22
P 0.21
C 0.04
LK 0.11
WK 0.12
A_Coef 0.08
LKG 0.23

Best Parameters: {'clf__bootstrap': False, 'clf__criterion': 'gini', 'clf__n_estimators': 200}

 [[13  1  0]
 [ 2 12  0]
 [ 0  0 14]]

               precision    recall  f1-score   support

        Kama       0.87      0.93      0.90        14
        Rosa       0.92      0.86      0.89        14
    Canadian       1.00      1.00      1.00        14

    accuracy                           0.93        42
   macro avg       0.93      0.93      0.93        42
weighted avg       0.93      0.93      0.93        42


Optimized Model

Model Name: DecisionTreeClassifier(max_depth=6, random_state=100)


Feature Importances
A 0.34
P 0.02
C 0.0
LK 0.0
WK 0.03
A_Coef 0.06
LKG 0.55

Best Parameters: {'clf__criterion': 'gini', 'clf__max_depth': 6}

 [[13  1  0]
 [ 2 12  0]
 [ 1  0 13]]

               precision    recall  f1-score   support

        Kama       0.81      0.93      0.87        14
        Rosa       0.92      0.86      0.89        14
    Canadian       1.00      0.93      0.96        14

    accuracy                           0.90        42
   macro avg       0.91      0.90      0.91        42
weighted avg       0.91      0.90      0.91        42

In [ ]:
 
