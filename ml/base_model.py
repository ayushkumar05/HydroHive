import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
# Define the Decision Tree classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = self.build_tree(X, y, max_depth=self.max_depth)

    def build_tree(self, X, y, max_depth=None):
        if len(set(y)) == 1:
            return {'label': y[0]}
        if max_depth == 0:
            return {'label': max(set(y), key=y.count)}
        feature_index, threshold = self.find_best_split(X, y)
        left_indices = [i for i in range(len(X)) if X[i][feature_index] <= threshold]
        right_indices = [i for i in range(len(X)) if X[i][feature_index] > threshold]
        left_tree = self.build_tree(X[left_indices], [y[i] for i in left_indices], max_depth=max_depth-1)
        right_tree = self.build_tree(X[right_indices], [y[i] for i in right_indices], max_depth=max_depth-1)
        return {'feature_index': feature_index, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def find_best_split(self, X, y):
        best_feature_index = None
        best_threshold = None
        best_gini_impurity = float('inf')
        for feature_index in range(len(X[0])):
            values = sorted(list(set([x[feature_index] for x in X])))
            for i in range(1, len(values)):
                threshold = (values[i - 1] + values[i]) / 2
                left_indices = [j for j in range(len(X)) if X[j][feature_index] <= threshold]
                right_indices = [j for j in range(len(X)) if X[j][feature_index] > threshold]
                gini_impurity = self.gini_impurity(y[left_indices], y[right_indices])
                if gini_impurity < best_gini_impurity:
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_gini_impurity = gini_impurity
        return best_feature_index, best_threshold

    def gini_impurity(self, y1, y2):
        n = len(y1) + len(y2)
        p1 = sum([1 for y in y1 if y == 1]) / len(y1) if len(y1) > 0 else 0
        p2 = sum([1 for y in y2 if y == 1]) / len(y2) if len(y2) > 0 else 0
        return (len(y1) / n) * (1 - p1 ** 2 - (1 - p1) ** 2) + (len(y2) / n) * (1 - p2 ** 2 - (1 - p2) ** 2)

    def predict_proba(self, X):
        node = self.tree_
        while 'label' not in node:
            if X[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['label']


# Define the Random Forest classifier
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            sample_indices = [i % len(X) for i in range(i, i + self.n_estimators)]
            X_train = X.iloc[sample_indices]
            y_train = [y[i] for i in sample_indices]
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_train, y_train)
            self.estimators_.append(tree)

    def predict_proba(self, X):
        predictions = []
        for tree in self.estimators_:
            predictions.append(tree.predict_proba(X)[0][1])
        return sum(predictions) / len(predictions)

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        #change max depth to 1 as 3 can make the person confused 
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        self.estimators_ = []
        n_samples, n_features = X.shape
        y_pred = 0.5 * pd.Series([0] * n_samples)

        for i in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residuals)
            self.estimators_.append(tree)
            update = tree.predict(X)
            y_pred += self.learning_rate * update

    def predict(self, X):
        y_pred = 0.5 * pd.Series([0] * X.shape[0])
        for tree in self.estimators_:
            update = tree.predict(X)
            y_pred += self.learning_rate * update
        return y_pred.apply(lambda x: 1 if x >= 0 else 0)

class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=200, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.coefs_ = []
        self.intercepts_ = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(set(y))
        self.coefs_ = []
        self.intercepts_ = []

        # One-hot encode the target variable
        y_onehot = pd.get_dummies(y).values

        # Initialize weights and biases for the input and hidden layers
        self.coefs_.append(np.random.randn(n_features, self.hidden_layer_sizes[0]))
        self.intercepts_.append(np.random.randn(self.hidden_layer_sizes[0]))

        for i in range(1, len(self.hidden_layer_sizes)):
            self.coefs_.append(np.random.randn(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i]))
            self.intercepts_.append(np.random.randn(self.hidden_layer_sizes[i]))

        # Initialize weights and biases for the output layer
        self.coefs_.append(np.random.randn(self.hidden_layer_sizes[-1], n_classes))
        self.intercepts_.append(np.random.randn(n_classes))

        # Train the network using backpropagation
        for iter in range(self.max_iter):
            z = [None] * (len(self.hidden_layer_sizes) + 1)
            a = [None] * (len(self.hidden_layer_sizes) + 1)
            a[0] = X

            # Forward pass
            for i in range(len(self.hidden_layer_sizes)):
                z[i+1] = np.dot(a[i], self.coefs_[i]) + self.inter



# Load the dataset
data = pd.read_csv('northern_bihar.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['Temperature', 'Rainfall', 'Water Level', 'Terrain']], data['Flood Risk'], test_size=0.2, random_state=42)

# Train multiple classifiers
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
]
predictions = []
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    predictions.append(y_pred)

# Combine the predictions using majority voting
# combined_pred = []
# for i in range(len(predictions[0])):
#     counts = {}
#     for pred in predictions:
#         if pred[i] not in counts:
#             counts[pred[i]] = 1
#         else:
#             counts[pred[i]] += 1
#     combined_pred.append(max(counts, key=counts.get))

'''
The weightages assigned to each terrain type may vary depending on various factors such as historical flood data, the frequency and severity of floods in each region, the topography of the region, and the density of population living in flood-prone areas. However, in the absence of such data, we can make certain assumptions based on the characteristics of each terrain type.

Himalayan Mountains: The Himalayan region is characterized by steep slopes and high altitude which can lead to flash floods due to heavy rainfall or snowmelt. Hence, this region is assigned a higher weightage of 0.3.

Northern Plains: The Northern plains are mostly flat and are drained by several rivers such as Ganga and Yamuna. These rivers are prone to flooding due to heavy rainfall and snowmelt in the Himalayan region. Hence, this region is assigned a weightage of 0.2.

Peninsular Plateaus: The Peninsular Plateaus are characterized by undulating topography and have a moderate rainfall. Floods in this region are mostly caused due to dam failures or flash floods. Hence, this region is assigned a weightage of 0.15.

Indian Deserts: The Indian deserts receive very little rainfall and are mostly arid. Floods in this region are rare and are mostly caused due to cloud bursts or flash floods. Hence, this region is assigned a weightage of 0.1.

Coastal Plains: The Coastal Plains are characterized by low-lying areas and are prone to flooding due to storm surges or heavy rainfall. Hence, this region is assigned a weightage of 0.2.

Islands: The Islands are also low-lying areas and are prone to flooding due to storm surges or sea-level rise. Hence, this region is assigned a weightage of 0.05.

It is important to note that these weightages are only indicative and may vary based on various factors such as the specific location, the data available
'''

# Define weightage for each model
weights = {
    "RandomForestClassifier": 0.3,
    "DecisionTreeClassifier": 0.2,
    "GradientBoostingClassifier": 0.3,
    "MLPClassifier": 0.2,
}

# Define terrain type weightage
terrain_weights = {
    "Himalayan Mountains": {"RandomForestClassifier": 0.1, "DecisionTreeClassifier": 0.1, "GradientBoostingClassifier": 0.3, "MLPClassifier": 0.5},
    "Northern Plains": {"RandomForestClassifier": 0.5, "DecisionTreeClassifier": 0.1, "GradientBoostingClassifier": 0.2, "MLPClassifier": 0.2},
    "Peninsular Plateaus": {"RandomForestClassifier": 0.3, "DecisionTreeClassifier": 0.2, "GradientBoostingClassifier": 0.2, "MLPClassifier": 0.3},
    "Indian Deserts": {"RandomForestClassifier": 0.7, "DecisionTreeClassifier": 0.1, "GradientBoostingClassifier": 0.1, "MLPClassifier": 0.1},
    "Coastal Plains": {"RandomForestClassifier": 0.2, "DecisionTreeClassifier": 0.3, "GradientBoostingClassifier": 0.3, "MLPClassifier": 0.2},
    "Islands": {"RandomForestClassifier": 0.1, "DecisionTreeClassifier": 0.4, "GradientBoostingClassifier": 0.4, "MLPClassifier": 0.1},
}

def get_terrain_type(latitude, longitude):
    # Fetch terrain data from API
    response = requests.get(f"https://api.opentopodata.org/v1/srtm30m?locations={latitude},{longitude}")
    data = response.json()

    # Extract terrain type
    elevation = data["results"][0]["elevation"]
    if elevation <= 100:
        terrain_type = "Coastal Plains"
    elif elevation <= 300:
        terrain_type = "Northern Plains"
    elif elevation <= 600:
        terrain_type = "Peninsular Plateaus"
    elif elevation <= 1500:
        terrain_type = "Indian Deserts"
    elif elevation <= 4000:
        terrain_type = "Himalayan Mountains"
    else:
        terrain_type = "Islands"
    
    return terrain_type

def predict_risk_level(latitude, longitude, terrain):
    # fetch data for the given location
    data = fetch_data(latitude, longitude)
    
    # preprocess the data
    data = preprocess_data(data)
    
    # load the models
    models = load_models()
    
    # predict the risk level using each model
    model_predictions = []
    for model in models:
        # get the prediction probabilities
        proba = model.predict_proba(data)[0]
        # himalyan mountains , northern plains, peninsulare plateaus, indian desserts, costal plains, islands
        # adjust the weightage of the prediction based on terrain
        if terrain == "mountain":
            weightage = 0.6 # mountainous terrain has higher flood risk, so give more weight to the prediction
        elif terrain == "coastal":
            weightage = 0.3 # coastal areas also have higher flood risk, but not as much as mountains
        else:
            weightage = 0.1 # other terrains have lower flood risk, so give less weight to the prediction
        
        # multiply the prediction probabilities with the weightage factor
        weighted_proba = proba * weightage
        
        # add the weighted prediction to the list
        model_predictions.append(weighted_proba)
    
    # combine the predictions from all models
    combined_proba = np.mean(model_predictions, axis=0)
    
    # get the final risk level prediction
    risk_level = np.argmax(combined_proba)
    
    # get a list of nearby towns with lower flood risk
    safe_towns = get_safe_towns(latitude, longitude)
    
    # return the risk level prediction and the list of safe towns
    return risk_level, safe_towns


# Evaluate the combined predictions
print(f"Combined Classifier Accuracy: {accuracy_score(y_test, combined_pred)}")
print(classification_report(y_test, combined_pred))

'''
DecisionTreeClassifier:
Strengths:
Can handle both categorical and numerical data
Non-parametric, so it doesn't make assumptions about the underlying distribution of the data
Easy to interpret and visualize the decision-making process
Weaknesses:

Can be prone to overfitting, especially if the tree is deep or the data is noisy
Sensitive to small variations in the training data, which can result in different trees being generated
For example, in predicting floods, DecisionTreeClassifier can be used to identify the most important features or predictors that lead to flooding in a certain location, such as high rainfall, low elevation, or proximity to a river. However, DecisionTreeClassifier may not be sufficient to capture the complex interactions between these features and may result in overfitting, especially if the dataset is small or the features are highly correlated.

RandomForestClassifier:
Strengths:
Combines multiple decision trees to reduce overfitting and improve generalization
Can handle both categorical and numerical data
Provides a measure of feature importance, which can help identify the most relevant predictors
Weaknesses:

Can be computationally expensive to train and evaluate, especially for large datasets or deep trees
Can be difficult to interpret and understand the decision-making process, especially if there are many trees in the forest
In terms of predicting floods, RandomForestClassifier can be used to build an ensemble of decision trees that capture different aspects of the data, such as the importance of rainfall, temperature, elevation, or proximity to a river. By combining the predictions of multiple trees, RandomForestClassifier can reduce overfitting and improve the accuracy of the predictions.

GradientBoostingClassifier:
Strengths:
Builds an ensemble of weak learners that are optimized to correct errors of previous learners
Can handle both categorical and numerical data
Often achieves high accuracy and can generalize well to new data
Weaknesses:

Can be computationally expensive to train and evaluate, especially for large datasets or deep trees
Can be prone to overfitting, especially if the learning rate or number of iterations is set too high
For predicting floods, GradientBoostingClassifier can be used to iteratively improve the accuracy of the model by building a sequence of trees that correct the errors of the previous trees. This can be especially useful if the data is noisy or there are complex interactions between the predictors that are difficult to capture with a single tree.

MLPClassifier:
Strengths:
Can capture complex non-linear relationships between the features and target variable
Can handle both categorical and numerical data
Can generalize well to new data if properly regularized
Weaknesses:

Can be computationally expensive to train and evaluate, especially for large datasets or deep networks
Can be prone to overfitting, especially if the network is too large or not properly regularized
In predicting floods, MLPClassifier can be used to capture the complex interactions between the predictors and the target variable, such as the effects of temperature, rainfall, and elevation on the risk of flooding. However, due to its complexity, MLPClassifier can be prone to overfitting and may require careful regularization to achieve good generalization.
'''