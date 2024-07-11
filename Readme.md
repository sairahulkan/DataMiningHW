# Titanic Survival Prediction and Analysis

This project demonstrates a comprehensive approach to data analysis and machine learning using the Titanic dataset. The main goal is to predict passenger survival using various machine learning algorithms and feature engineering techniques.

## Features

- **Data Preprocessing**:
  - Handling missing values.
  - Feature engineering including creation of new features like `Title`, `Age*Class`, and `Deck`.
  - Encoding categorical variables.
  - Scaling and normalization of data.

- **Exploratory Data Analysis (EDA)**:
  - Visualization of data using Seaborn and Matplotlib.
  - Analysis of feature distributions and their relationships with survival.

- **Machine Learning Models**:
  - Implementing and evaluating various classification algorithms:
    - Logistic Regression
    - Support Vector Machines (SVM)
    - K-Nearest Neighbors (KNN)
    - Gaussian Naive Bayes
    - Perceptron
    - Linear SVC
    - Stochastic Gradient Descent (SGD)
    - Decision Tree
    - Random Forest

## Technologies Used

- **Python**: Programming language.
- **Jupyter Notebook**: Interactive environment for data analysis and visualization.
- **Pandas**: Data manipulation and analysis.
- **Numpy**: Numerical computing library.
- **Matplotlib/Seaborn**: Data visualization libraries.
- **Scikit-learn**: Machine learning library.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sairahulkan/TitanicSurvivalPrediction.git
    cd TitanicSurvivalPrediction
    ```

2. Install the necessary Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Start Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4. Open the Jupyter notebook `TitanicSurvivalPrediction.ipynb` to explore the analysis and model training process.

### Data

- **train.csv**: Training dataset.
- **test.csv**: Test dataset.

## Usage

1. **Data Preprocessing and EDA**:
   - Open `TitanicSurvivalPrediction.ipynb` in Jupyter Notebook.
   - Run the cells sequentially to preprocess the data and perform exploratory data analysis.

2. **Model Training and Evaluation**:
   - Continue running the cells in `TitanicSurvivalPrediction.ipynb` to train various machine learning models.
   - Evaluate the models using cross-validation and compare their performance.
