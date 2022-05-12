# Comparison of various machine learning alogrithms
## Scikit-learn
### Scikit-learn — Choosing the right estimator (original image has names clickable)
![Choosing the right estimator](https://scikit-learn.org/stable/_static/ml_map.png)

### Some commonly used algorithms

[Scikit-learn awesome documentation (supervised, semi-supervised, unsupervised, model selection and evaluation, etc.)](https://scikit-learn.org/stable/user_guide.html).  
Source: [link](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)


Here is a table with some commonly used Scikit-learn algorithms:

| **Algorithm**               | **Model type**               | **Type**       | **Supervised?** | **Semi-supervised?** | **Unsupervised?** | **Reinforcement?** | **Can do batch (incremental) training?** |
|-----------------------------|------------------------------|----------------|:---------------:|:--------------------:|:-----------------:|:------------------:|:----------------------------------------:|
| SGDClassifier               | Linear                       | Classification |        ✅        |                      |                   |                    |                     ✅                    |
| SGDRegressor                | Linear                       | Regression     |        ✅        |                      |                   |                    |                     ✅                    |
| Perceptron                  | Linear                       | Classification |        ✅        |                      |                   |                    |                     ✅                    |
| PassiveAggressiveClassifier | Linear                       | Classification |        ✅        |                      |                   |                    |                     ✅                    |
| PassiveAggressiveRegressor  | Linear                       | Regression     |        ✅        |                      |                   |                    |                     ✅                    |
| SVC                         | SVM                          | Classification |        ✅        |                      |                   |                    |                                          |
| LinearSVC                   | SVM                          | Classification |        ✅        |                      |                   |                    |                                          |
| SVR                         | SVM                          | Regression     |        ✅        |                      |                   |                    |                                          |
| LinearSVR                   | SVM                          | Regression     |        ✅        |                      |                   |                    |                                          |
| BernoulliNB                 | Naive Bayes                  | Classification |        ✅        |                      |                   |                    |                     ✅                    |
| MultinomialNB               | Naive Bayes                  | Classification |        ✅        |                      |                   |                    |                     ✅                    |
| GaussianNB                  | Naive Bayes                  | Classification |        ✅        |                      |                   |                    |                     ✅                    |
| DecisionTreeClassifier      | Decision Tree                | Classification |        ✅        |                      |                   |                    |                                          |
| AdaBoostClassifier          | Ensemble                     | Classification |        ✅        |                      |                   |                    |                                          |
| DecisionTreeRegressor       | Decision Tree                | Regression     |        ✅        |                      |                   |                    |                                          |
| GradientBoostingClassifier  | Ensemble                     | Classification |        ✅        |                      |                   |                    |                                          |
| GradientBoostingRegressor   | Ensemble                     | Regression     |        ✅        |                      |                   |                    |                                          |
| MLPClassifier               | Neural Network               | Classification |        ✅        |                      |                   |                    |                                          |
| MLPRegressor                | Neural Network               | Regression     |        ✅        |                      |                   |                    |                                          |
| NearestNeighbors            | Nearest Neighbors            | Learner        |                 |                      |         ✅         |                    |                                          |
| KNeighborsClassifier        | Nearest Neighbors            | Classification |                 |                      |         ✅         |                    |                                          |
| KNeighborsRegressor         | Nearest Neighbors            | Regression     |                 |                      |         ✅         |                    |                                          |
| BernoulliRBM                | Restricted Boltzmann Machine | Classification |                 |                      |         ✅         |                    |                                          |
| SelfTrainingClassifier      | Semi-supervised              | Classification |                 |           ✅          |                   |                    |                                          |
| LabelPropagation            | Semi-supervised              |                |                 |           ✅          |                   |                    |                                          |
| LabelSpreading              | Semi-supervised              |                |                 |           ✅          |                   |                    |                                          |
| KMeans                      | Clustering                   | Clustering     |                 |                      |         ✅         |                    |                                          |
| MiniBatchKMeans             | Clustering                   | Clustering     |                 |                      |         ✅         |                    |                     ✅                    |



