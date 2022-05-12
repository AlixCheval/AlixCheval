# Machine learning useful notes
Right now, it is considered that any problem a human can do in 1 second can be done by a machine learning program.

## Ressources 
### Learning
- [Machine learning mastery](https://machinelearningmastery.com/start-here/)
- [Deep learning — Wikipedia article](https://en.wikipedia.org/wiki/Deep_learning)

### Datasets
Some datasets are considered the "hello world" of datasets, such as [MNIST digit dataset](http://yann.lecun.com/exdb/mnist/) or the [UCI Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris). 
- [Subreddit datasets](https://www.reddit.com/r/datasets/)
- [Kaggle datasets](https://www.kaggle.com/datasets)
- [Academic torrents — 127.15TB of research data](https://academictorrents.com/)
- [Wikipedia dataset](https://en.wikipedia.org/wiki/Wikipedia:Database_download)
- [Open Data Network](https://www.opendatanetwork.com/)
- [Nasa Earth science datasets](https://earthdata.nasa.gov/)
- [Nasa space datasets](https://pds.nasa.gov/datasearch/data-search/)
- [AWS public datasets](https://registry.opendata.aws/)
- [Google Cloud public datasets](https://cloud.google.com/bigquery/public-data/)
- [QuantConnect datasets (for trading)](https://www.quantconnect.com/datasets)
- [UCI Machine Learning datasets](https://archive.ics.uci.edu/ml/index.php)


## Pipeline
It is important to have a machine learning pipeline. It is a way to codify, automate and industrialize the workflow of a machine learning project.

### Global pipeline

1. Retrieve data
2. Exploratory Data Analysis (some data can be corrupted, missing, useless, wrong, etc.). It helps understand avaiable data better 
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
- Seeks to maximize the notion of cumulative reward (there is a points system)
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
cf. [This Markdown file](./ml_comparison.md)

## Clasical machine learning — Baseline model
Before classical machine learning, people had to provide data + feature extraction + the algorithm to the solution.  
Example: to denoise an image, someone had to have the images, extract the features related to image denoising, and provide the algorithm responsibles of the denoising. 

**With  classical machine learning, the model needs data + the features. The algorithm find itself the agorithmic solution.**

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

However, sometimes a classical model will not just be enough, because it is too simple and the reality of the problem is more complex.

## Deep learning
With classical machine learning, we need to provide the data and the features to the model.  
**With deep learning, only the data are needed, the algorithm find itself the features related to the data** (example: to identify a face, it learns that it is made of 2 eyes, a mouth, etc.).
It is possible to have hybrid models (example: RNN + CNN)

- [State of the Art in deep learning architectures](https://paperswithcode.com/sota)
- [Model zoo (examples) of deep learning codes and models](https://modelzoo.co/) 

### Downside of deep learning
- Computically expensive, needs lots of ressources/time/money
- Needs a lot of data
- Mainly empirical, it just works
- Bad explainability, works as a black box, [although it tends to change](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence) 
- Prone to overfit (explained later)
- Therefore, vulnerable to  attacks, [such as adversarial machine learning](https://en.wikipedia.org/wiki/Adversarial_machine_learning)

### Neural network classes
- MLP (Multilayer Perceptron)
    - Kind of a baseline model but in deep learning, can be used with images, text, time series or other data
    - For classification & regression
    - Used with tabular datasets
- CNN (Convolutional Neural Network)
    - Often compared to the way the brain achieves vision processing in living organisms
    - Feed forward
    - Most commonly applied to analyze visual imagery
    - [A list of applications](https://en.wikipedia.org/wiki/Convolutional_neural_network#Applications)
    - Examples of architectures: VGG16, VGG19, ResNet, Densenet, Inception, Squeeze-and-Excitation Networks, EfficientNet
    - [More here](https://paperswithcode.com/methods/category/convolutional-neural-networks)
    - [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan) can be used to generate images.
- RNN (Recurrent Neural Network)
    - Theoretically Turing complete
    - Recursive (i.e. not feed forward)
    - Applications: speech recognitions,  handwriting recognition, machine translation, or automatic image captioning (when combined with CNNs)
    - **Not** suitable for tabular or image data
    - Example of architectures: LSTM, GRU  
    - [TensorFlow time series forecasting tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
    - [More here](https://paperswithcode.com/methods/category/recurrent-neural-networks)
- [Deep reinforcement learning](https://en.wikipedia.org/wiki/Deep_reinforcement_learning) (combines reinforcement learning with deep learning)
    - Used for a diverse set of applications including but not limited to robotics, video games, natural language processing, computer vision, education, transportation, finance and healthcare ([source](https://en.wikipedia.org/wiki/Deep_reinforcement_learning))
    - [Gym — toolkit for developing and comparing reinforcement learning algorithms](https://gym.openai.com/docs/)
    - [TensorFlow introduction](https://www.tensorflow.org/agents/tutorials/0_intro_rl), [Pytorch introduction](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), [OpenAI tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
