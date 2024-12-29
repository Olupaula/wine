# Cluster Analysis of Branded Wine
Wine is great drink for many people. it remains true that our world cannot be demand for it. It therefore makes good sense if wine can be categorized into a number of classes by a seller or even the consumer of the wine. This can be achieved using a suitable Machine Learning clustering model.

## Data Source
UCL Machine Learning Repository

## Data Features and Target
The dataset had 13 features and a target. The features are:
1. Alcohol
2. Malic acid
3.Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

The target is: class

## Methods
This work uses the followinf unsupervised learning techniques:
1. K-Means
2. Busecting K-Means
3. Mini-Batch K-Means
These algorithm are used to cluster wines with 13 features. Each feature has its label, however the interest in this work is to compare the three unsupervised learning methods stated above. 

## Models Performances Evaluation
1. Inertial: here, the best model is the one with lowest inertia or sum of square error.
2. Accuracy Score: using the pre-existing labels, the accuracy of clustering of the three methods. The higher the accuracy, the better the model.

## Best Model
of the three models were compared and the K-Means had the best performance, since its inertia was the lowest and it had the highest accuracy score.




