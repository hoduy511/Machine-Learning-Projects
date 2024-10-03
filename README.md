# Machine Learning Notebooks Project

## Project Description
This project contains a collection of notebooks that implement various Machine Learning problems, categorized into two main groups: supervised learning and unsupervised learning. These problems include classification, regression, and clustering, with each problem solved using different algorithms to provide a comprehensive understanding of Machine Learning techniques and tools.

## Directory Structure

### [`notebooks/`](notebooks)
Contains all the notebooks of the project, divided into two main groups:

#### 1. [`supervised/`](notebooks/supervised)
Includes supervised learning problems, with two main types: **classification** and **regression**.

##### [`classification/`](notebooks/supervised/classification)
- **[`decision_tree/`](notebooks/supervised/classification/decison_tree)**: 
  - [`breast_cancer_decision_tree_classifier.ipynb`](notebooks/supervised/classification/decison_tree/breast_cancer_decision_tree_classifier.ipynb): Decision tree classification for breast cancer data.
  - [`iris_decision_tree_classifier.ipynb`](notebooks/supervised/classification/decison_tree/iris_decision_tree_classifier.ipynb): Iris flower classification using decision tree.
  - [`wine_decision_tree_classifier.ipynb`](notebooks/supervised/classification/decison_tree/wine_decision_tree_classifier.ipynb): Wine classification using decision tree.

- **[`knn/`](notebooks/supervised/classification/knn)**:
  - [`digit_recognition.ipynb`](notebooks/supervised/classification/knn/digit_recognition.ipynb): Handwritten digit recognition using K-Nearest Neighbors (KNN).

- **[`model_comparison_classification/`](notebooks/supervised/classification/model_comparison_classification)**:
  - [`classifier_comparison.ipynb`](notebooks/supervised/classification/model_comparison_classification/classifier_comparison.ipynb): Comparison of different classification models on a specific dataset.
  - [`digit_classifier_comparison.ipynb`](notebooks/supervised/classification/model_comparison_classification/digit_classifier_comparison.ipynb): Comparison of different classification models on the digit dataset.
  - [`spam_classifier_comparison.ipynb`](notebooks/supervised/classification/model_comparison_classification/spam_classifier_comparison.ipynb): Comparison of different models for spam classification.

- **[`neural_network/`](notebooks/supervised/classification/neural_network)**:
  - **[`mlp_classifier/`](notebooks/supervised/classification/neural_network/mlp_classifier)**:
    - [`diabetes_mlp_classifier.ipynb`](notebooks/supervised/classification/neural_network/mlp_classifier/diabetes_mlp_classifier.ipynb): Using MLP network for diabetes classification.
    - [`iris_mlp_classifier.ipynb`](notebooks/supervised/classification/neural_network/mlp_classifier/iris_mlp_classifier.ipynb): Using MLP network for Iris flower classification.
  
  - **[`perceptron/`](notebooks/supervised/classification/neural_network/perceptron)**:
    - [`__pycache__/`](notebooks/supervised/classification/neural_network/perceptron/__pycache__): Contains compiled Python files for Perceptron implementation.
    - [`diabetes_perceptron.ipynb`](notebooks/supervised/classification/neural_network/perceptron/diabetes_perceptron.ipynb): Diabetes classification using Perceptron.
    - [`iris_perceptron.ipynb`](notebooks/supervised/classification/neural_network/perceptron/iris_perceptron.ipynb): Iris flower classification using Perceptron.
    - [`perceptron_from_scratch.py`](notebooks/supervised/classification/neural_network/perceptron/perceptron_from_scratch.py): Python file implementing Perceptron from scratch without using specialized libraries.
    - [`perceptron_from_scratch_demo.ipynb`](notebooks/supervised/classification/neural_network/perceptron/perceptron_from_scratch_demo.ipynb): Demo implementation of Perceptron from scratch.
    - [`spam_perceptron.ipynb`](notebooks/supervised/classification/neural_network/perceptron/spam_perceptron.ipynb): Spam classification using Perceptron.

##### [`regression/`](notebooks/supervised/regression)
- **[`linear/`](notebooks/supervised/regression/linear)**:
  - [`advertising.ipynb`](notebooks/supervised/regression/linear/advertising.ipynb): Predicting advertising effectiveness using linear regression.
  - [`boston_house_price.ipynb`](notebooks/supervised/regression/linear/boston_house_price.ipynb): Predicting Boston house prices using linear regression.
  - [`linear_regression_from_scratch.ipynb`](notebooks/supervised/regression/linear/linear_regression_from_scratch.ipynb): Linear regression implemented from scratch without using specialized libraries.
  - [`student_performance.ipynb`](notebooks/supervised/regression/linear/student_performance.ipynb): Predicting student performance using linear regression.

#### 2. [`unsupervised/`](notebooks/unsupervised)
Contains unsupervised learning problems, including clustering algorithms.

##### [`clustering/`](notebooks/unsupervised/clustering)
- **[`dbscan/`](notebooks/unsupervised/clustering/dbscan)**:
  - [`custom_dbscan_implementation.ipynb`](notebooks/unsupervised/clustering/dbscan/custom_dbscan_implementation.ipynb): Custom implementation of DBSCAN from scratch without using specialized libraries.
  - [`fresh_milk_clustering.ipynb`](notebooks/unsupervised/clustering/dbscan/fresh_milk_clustering.ipynb): Clustering the `Fresh` and `Milk` dataset using DBSCAN.

- **[`kmeans/`](notebooks/unsupervised/clustering/kmeans)**:
  - [`kmeans_from_scratch.ipynb`](notebooks/unsupervised/clustering/kmeans/kmeans_from_scratch.ipynb): Implementation of the K-means algorithm from scratch without using specialized libraries.
  - [`news_group_clustering.ipynb`](notebooks/unsupervised/clustering/kmeans/news_group_clustering.ipynb): Clustering news articles into topic groups using K-means.

### [`data/`](data)
Contains datasets used in the project.

#### 1. [`classification/`](data/classification)
- `SMSSpamCollection`: Dataset for spam classification.
- `and_gate_datasets.csv`: Dataset for AND gate classification.

#### 2. [`regression/`](data/regression)
- `advertising.csv`: Dataset for advertising effectiveness prediction.
- `housing.csv`: Dataset for housing prices.
- `student_performance.csv`: Dataset for student performance prediction.

## Usage

To use the notebooks in this project, follow these steps:

1. **Clone the Repository**: Clone the repository to your local machine using:
   ```bash
   git clone https://github.com/hoduy511/Machine-Learning-Projects.git
   cd Machine-Learning-Projects
   ```

2. **Set Up the Environment**: It is recommended to use a virtual environment to manage dependencies. You can create and activate a virtual environment using the following commands:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Required Libraries**: Install the required libraries listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Notebooks**: Open Jupyter Notebook and navigate to the `notebooks` directory to run the desired notebooks:
   ```bash
   jupyter notebook
   ```

## Requirements

This project requires the following libraries:

- Python 3.10.12
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

You can find a complete list of libraries in the `requirements.txt` file, which can be installed using pip.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contributing

Contributions to this project are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch-name
   ```

3. Make your changes and commit them:
   ```bash
   git commit -m "Add a meaningful commit message"
   ```

4. Push your branch:
   ```bash
   git push origin feature-branch-name
   ```

5. Open a pull request.
