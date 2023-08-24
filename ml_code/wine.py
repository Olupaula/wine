import joblib
import pandas as pd
import numpy as np
from scipy.stats import bartlett

from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('../dataset/wine.data', sep=',')
print(data.head())

data.columns = [
    'class',
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
]

print(data)

data['class'] = data['class'].apply(
    lambda label:
    0 if label == 1
    else 1 if label == 2
    else 2
)

columns_unique_values = data.apply(lambda value: value.unique())
print(columns_unique_values)

# checking for missing data
columns = data.columns
print('Any missing data?')

for column in columns:
    if data[column].isna().any():
        print()
        print(column + ':', data[column].isna().any())
        data[column].fillna(data[column].mode(dropna=True))
    else:
        print('no missing data')

# Inspecting Variances
print(data.mean())
print('sd:')
print(data.var())

# Checking for homogeneity of variance - a requirement for running K-means
homogeneity_of_variance = bartlett(data['Malic acid'], data['Alcohol'])
print(homogeneity_of_variance)

homogeneity_of_variance = bartlett( data['Alcohol'], data['Flavanoids'])
print(homogeneity_of_variance)

homogeneity_of_variance = bartlett(data['Malic acid'], data['Flavanoids'])
print(homogeneity_of_variance)

homogeneity_of_variance = bartlett(data['Hue'], data['Ash'])
print(homogeneity_of_variance)


def equalize_target_cat(data_, target_name):
    """
    This function clusters equiprobable i.e. making the prior probability for all the clusters the same
    It first checks for a balance in categories of the target variable.
    After checking, if the classes are unbalanced, it removes some data from the
    classes (clusters) with more data and return the unique values counts of the target variable or
    However, if the classes are balanced, it returns the original data and the values counts of the target variable.

    **Note**:  Encode the target variable before supplying it to this function.
    :param data_: the dataframe to be used.
    :param target_name: target_name: the name of the column for the target variable.
    :return: a dictionary containing the data and value_counts() of the target variable.
    """
    print(data_.columns)
    targets_unique_values = data_[target_name].value_counts()
    print(targets_unique_values)
    # print(targets_unique_values)

    # whether or note targets classes are equal
    equal_targets = False
    if max(targets_unique_values) == min(targets_unique_values):
        equal_targets = True

    data_list = []  # a list of the dataframes, each dataframe having a unique label

    if not equal_targets:
        for i in range(len(targets_unique_values)):
            # print(len(targets_unique_values))
            print(min(targets_unique_values))
            partial_data = data_[data_[target_name] == i].sample(min(targets_unique_values), random_state=9994)
            data_list.append(partial_data)

        new_data = pd.concat(data_list)
        return {'value_counts': new_data[target_name].value_counts(), 'data': new_data}
    else:
        # print('Note: Target has equal categories')
        return {'value_counts': data_[target_name].value_counts(), 'data': data_}


# making the classes equal probable.
equalized_data = equalize_target_cat(data, 'class')['data']

y = equalized_data['class']
x = equalized_data.drop(labels=['class'], axis=1)
x = x[['Ash', 'Hue']]
x_columns = x.columns

# visualization of segmentation variables
sns.pairplot(x)
plt.show()


def model_result(model, name, x_, y_):
    print('_' * 50)
    print(model)
    print('_' * 50)
    model.fit(x)
    y_predicted_ = pd.DataFrame(model.predict(x_))
    accuracy = round(accuracy_score(y_predicted_, y_),4)
    centroids_ = model.cluster_centers_
    inertia_ = model.inertia_
    confusion_matrix_ = pd.crosstab(np.array(y_predicted_).T, np.array(y_).T, rownames=['predicted'], colnames=['actual'])
    print('inertia =', inertia_)
    print('accuracy =', accuracy)
    print('confusion_matrix')
    print(confusion_matrix_)
    if accuracy >= 0.65:
        joblib.dump(model, '../models/'+name+'.joblib')

    else:
        print(name+' not saved, because accuracy was'+str(round(accuracy, 4))+' < 0.70')

    return {'inertia': inertia_,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix_,
            'centroids': centroids_,
            'y_predicted': y_predicted_,
            'x': x_}


# Fitting the models
# 1) K-means
model1 = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    random_state=123
)
# To match with the labels for the clusters created by K-means
y_k = equalized_data['class'].apply(
    lambda label:
    0 if label == 0
    else 1 if label == 2
    else 2
)
model1_result = model_result(model1, 'K-Means', x, y_k)

# 2) Bisecting K-Means
model2 = BisectingKMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    random_state=123,
    bisecting_strategy='largest_cluster',
    max_iter=1000,
    algorithm='elkan'
)
# y for bisecting k-means
y_bs = equalized_data['class'].apply(
    lambda label:
    0 if label == 2
    else 1 if label == 0
    else 2
)
model2_result = model_result(model2, 'Bisecting K-Means', x, y_bs)

# 3) Mini-Batch K-Means
# y for mini batch k-means
y_mb = equalized_data['class'].apply(
    lambda label:
    0 if label == 1
    else 1 if label == 2
    else 2
)

model3 = MiniBatchKMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    batch_size=20,
    max_no_improvement=200,
    random_state=123,
)

model3_result = model_result(model3, 'Mini-Batch K-Means', x, y_mb)


# The scatter diagrams for each model
# The scatter diagram for model1
centroids = model1_result['centroids']  # the centroids to plot on the scatter diagram
y_predicted = model1_result['y_predicted']  # the predicted labels for each dataset
x = model1_result['x']
plt.scatter(x.iloc[:, 0],
            x.iloc[:, 1],
            c=y_predicted,
            s=20,
            cmap='copper',
            alpha=0.9)  # plotting the scatter diagram
plt.title('Plot of Clustering using K-Means')
plt.xlabel(x_columns[0])
plt.ylabel(x_columns[1])
plt.scatter(centroids[:, 0], centroids[:, 1], c=[0, 1, 2], cmap='copper', s=200)  # plotting the centroids
plt.show()

# The scatter diagram for model2
centroids = model2_result['centroids']  # the centroids to plot on the scatter diagram
y_predicted = model2_result['y_predicted']  # the predicted labels for each dataset
x = model2_result['x']
plt.scatter(x.iloc[:, 0],
            x.iloc[:, 1],
            c=y_predicted,
            s=20,
            cmap='copper',
            alpha=0.9)  # plotting the scatter diagram
plt.title('Plot of Clustering using Bisecting K-Means')
plt.xlabel(x_columns[0])
plt.ylabel(x_columns[1])
plt.scatter(centroids[:, 0], centroids[:, 1], c=[0, 1, 2], cmap='copper', s=200)  # plotting the centroids
plt.show()

# The scatter diagram for model3
centroids = model3_result['centroids']  # the centroids to plot on the scatter diagram
y_predicted = model3_result['y_predicted']  # the predicted labels for each dataset
x = model3_result['x']
plt.scatter(x.iloc[:, 0],
            x.iloc[:, 1],
            c=y_predicted,
            s=20, cmap='copper',
            alpha=0.9)  # plotting the scatter diagram
plt.title('Plot of Clustering using Mini-Batch K-Means')
plt.xlabel(x_columns[0])
plt.ylabel(x_columns[1])
plt.scatter(centroids[:, 0], centroids[:, 1], c=[0, 1, 2], cmap='copper', s=200)  # plotting the centroids
plt.show()
