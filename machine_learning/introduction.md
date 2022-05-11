# Machine learning useful notes

## Pipeline
### Global pipeline

1. Retrieve data
2. Exploratory Data Analysis (some data can be corrupted, missing, useless, wrong, etc.) 
3. Chose a model (a baseline model or a deep learning one). Do not pick a deep learning one if a baseline is sufficient!
4. Fit the model
5. Evaluate the model performances with KPIs (e.g. accuracy, precision, recall, F1-score, ROC curve). If the performances are bad, go back to step 1 if possible, or 2 if not (example: can't have more data)
    - Usually, 3 levels of performance: trivial < human performances < ready to deploy performances
7. Put model into production

### Pipeline challenges
- Data collection
- Data cleaning
- Feature extraction (labelling, dimensionality reduction)
- Model validation and understanding (it is important to know what the model is understanding and what its missing)
- Visualisation


## Baseline model
### Why start with a baseline model
- Faster training
- Better studied: harder to add bugs, easier to find flaws (e.g. bias) in data
- Ressources light
- Helps understand which classes are harder to separate
- Finds more easily what type of signal/feature the model detects
- Finds more easily what signal/feature the model is missing

### Example of baseline models
- Linear Regression
- Logistic Regression
- K-means clustering
- K-nearest neighbors
- Support Vector Machines
- Random Forest
- [Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- Fine tuning a [ResNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet) or an [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3)

## Types of ML
- Supervised learning 
- Semi-supervised learning
- Unsupervised learning
- Reinforcement learning

### Supervised learning
- Use labels attached to the data. The labels are defined by a human
- Must be careful that data are correctly labeled

### Semi-supervised learning
- Small amount of labeled data with a large amount of unlabeled data during training
- With a small amount of labeled data, the model can produce considerable improvement in learning accuracy

### Unsupervised learning
- Learns patterns from untagged data

### Reinforcement learning
- Goal-oriented algorithms
- Seeks to maximize the notion of cumulative reward
- No need in labelled input/output pairs
- No need in sub-optimal actions to be explicitly corrected
- No training dataset, learns from its experience.

### Various KPIs
- Loss
- Accuracy
- Precision
- Recall
- F1-score
- ROC curve
- K-fold cross validation

### Table of various algorithms

// TODO, explain their type of learning (supervised or not, etc.), their application (ex: classification, regression), if "out of core" is possible
