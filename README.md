
# Instacart Customer Segmentation & Reorder Prediction System

**Machine Learning Lab Mini-Project | 24CS01TP0501**

A machine learning project that segments grocery shoppers based on their purchasing behaviour and predicts which previously purchased products they are likely to reorder. The project uses the Instacart Market Basket Analysis dataset and includes an interactive Streamlit application for exploring customer segments and model results.

## Project Overview

The project combines unsupervised learning and supervised machine learning to understand grocery shopping patterns and predict repeat purchases.

- **Customer Segmentation:** Groups customers according to their ordering frequency, basket size, product preferences, and reorder behaviour.
- **Reorder Prediction:** Predicts whether a customer will reorder a previously purchased product.
- **Model Comparison:** Evaluates multiple classification algorithms using appropriate performance metrics.
- **Interactive Dashboard:** Presents the analysis, customer segments, model comparisons, and visualisations through Streamlit.
- **Data Analysis:** Includes exploratory data analysis, statistical summaries, missing-value checks, duplicate detection, outlier analysis, and visualisations.

## Lab Experiments Covered

| Experiment | Implementation |
|---|---|
| Experiment 2 - Exploratory Data Analysis | `src/eda.py` |
| Experiment 4 - k-Nearest Neighbours | `src/models.py` |
| Experiment 5 - Decision Tree | `src/models.py` |
| Experiment 7 - K-Means and Hierarchical Clustering | `src/segmentation.py` |
| Experiment 8 - Random Forest | `src/models.py` |
| Experiment 10 - Machine Learning Mini-Project | Complete project workflow |

## Dataset

**Instacart Market Basket Analysis**

The project uses the Instacart Online Grocery Shopping Dataset 2017, available through Kaggle.

- Approximately 3.4 million orders
- Approximately 200,000 users
- Product, aisle, and department information
- Prior order history and reordered-product labels

Dataset source: [Instacart Market Basket Analysis - Kaggle](https://www.kaggle.com/competitions/instacart-market-basket-analysis)

### Required Files

Download the competition dataset after accepting its rules and place the following files inside the `data/` directory:

```text
data/
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
├── products.csv
├── aisles.csv
└── departments.csv
```

The project also supports compressed CSV files where supported by the data loader.

## Project Structure

```text
Machine-Learning-Project/
│
├── app.py
├── run_pipeline.py
├── make_synthetic_data.py
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── eda.py
│   ├── segmentation.py
│   └── models.py
│
├── notebooks/
│   └── Jupyter / Colab workflow
│
├── data/
│   └── Instacart CSV files
│
├── models/
│   └── Saved model files
│
└── reports/
    ├── model_comparison.csv
    ├── Evaluation tables
    └── Generated figures
```

Generated files depend on the pipeline configuration and execution.

## Technologies Used

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Clustering:** K-Means, Hierarchical Clustering
- **Classification:** Logistic Regression, k-NN, Decision Tree, Random Forest, AdaBoost
- **Application:** Streamlit
- **Development Environment:** VS Code, Jupyter Notebook / Google Colab

## Methodology

### 1. Exploratory Data Analysis

The EDA module examines the structure and quality of the dataset through:

- Head, tail, shape, and descriptive statistics
- Missing-value and duplicate checks
- IQR-based outlier analysis
- Univariate, bivariate, and multivariate visualisations
- Analysis of ordering patterns and product reorders

Implementation: `src/eda.py`

### 2. Customer Segmentation

Each customer is represented by behavioural features such as:

- Number of orders and days between orders
- Average basket size
- Reorder ratio
- Number of distinct products purchased
- Organic, produce, and snack purchase shares
- Preferred order hour and weekend-order share

Skewed numerical features are log-transformed where appropriate and standardised before clustering.

**Clustering workflow:**

1. Prepare customer-level behavioural features.
2. Apply feature transformations and standardisation.
3. Evaluate K-Means cluster counts from 3 to 6 using silhouette scores.
4. Select a suitable cluster count based on the evaluation.
5. Apply Ward hierarchical clustering to a sample of 5,000 customers.
6. Compare the clustering results using the Adjusted Rand Index.
7. Assign descriptive names to segments based on their distinguishing characteristics.

Implementation: `src/segmentation.py`

### 3. Reorder Prediction

The classification task predicts whether a customer will reorder a product they have purchased previously.

Each candidate record represents a customer-product pair.

- **Target:** Whether the product appears in the customer's target training order.
- **Historical data:** Previous purchases from the `prior` orders.
- **Customer features:** Ordering frequency and other behavioural statistics.
- **Product features:** Product-level historical statistics.
- **Customer-product features:** Previous purchase count, reorder rate, orders since the last purchase, and cart position.
- **Context features:** Relevant information about the target order.
- **Segmentation feature:** The customer's assigned segment.

The classification algorithms evaluated are:

1. Logistic Regression
2. k-Nearest Neighbours (k-NN)
3. Decision Tree
4. Random Forest
5. AdaBoost
6. Majority-class baseline

Implementation: `src/models.py`

### 4. Model Evaluation

The models are evaluated using multiple metrics rather than accuracy alone.

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted reorders that are actual reorders |
| Recall | Proportion of actual reorders correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to distinguish between the two classes across thresholds |
| Average Precision | Summary of the precision-recall curve |
| Confusion Matrix | Breakdown of correct and incorrect classifications |

The generated model comparison results are saved in `reports/model_comparison.csv`.

## Preventing Data Leakage

Data leakage can make a machine learning model appear more accurate than it really is. The project applies the following controls:

- **User-level split:** Customers are split into training and testing sets before model development, using an 80:20 ratio.
- **Historical features:** Features are derived from prior orders, while target labels come from the designated training order.
- **Training-only fitting:** The scaler, K-Means model, and product-level statistics are fitted using training customers only.
- **Grouped cross-validation:** `RandomizedSearchCV` and `GroupKFold` are used for hyperparameter tuning, with customers as groups.
- **Validation-based threshold:** The classification threshold is selected using a validation split of training customers, not the test set.

These measures help produce a more reliable estimate of how the models generalise to unseen customers.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- pip
- Kaggle account for downloading the real dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kimaya0310/Machine-Learning-Project.git
cd Machine-Learning-Project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

Download the Instacart Market Basket Analysis dataset from Kaggle and place the required CSV files in the `data/` directory.

### Step 4: Run the Machine Learning Pipeline

```bash
python run_pipeline.py --data data --users 20000
```

The `--users` parameter controls the number of customers sampled for the workflow.

- `--users 20000`: Processes a sample of 20,000 customers.
- `--users 10000`: Processes a smaller sample.
- `--users 0`: Uses all available customers, subject to memory and computational requirements.
- `--fast`: Reduces the hyperparameter tuning budget for quicker experimentation.

Execution time depends on the computer, dataset size, and tuning configuration.

### Step 5: Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to explore the application.

## Running Without the Real Dataset

A synthetic dataset can be generated to test the workflow without downloading the Kaggle files.

```bash
python make_synthetic_data.py --users 4000
python run_pipeline.py --data data_synthetic --users 4000 --fast
streamlit run app.py
```

**Important:** Synthetic data is artificially generated for demonstration and testing. Its evaluation metrics must not be reported as results obtained from the real Instacart dataset.

## Google Colab

The workflow can also be executed in Google Colab.

1. Upload the project folder or ZIP archive to Colab.
2. Install the required dependencies.
3. Upload or otherwise make the real dataset available.
4. Run the pipeline.

Example commands:

```python
!pip install -r requirements.txt
%cd /content/Machine-Learning-Project
!python run_pipeline.py --data data --users 10000
```

The Jupyter notebook in `notebooks/` provides a step-by-step version of the workflow.

## Limitations

- **Class imbalance:** Only around 10–15% of candidate customer-product pairs may be positive reorder cases. Accuracy alone can therefore be misleading.
- **Computational cost:** Training and tuning on millions of candidate pairs can require substantial time and memory.
- **k-NN subsampling:** k-NN is trained on a 40,000-row subsample because of its computational cost. Other classifiers may use larger subsamples, as documented in the generated model comparison report.
- **Weekend assumption:** The dataset's day-of-week mapping is treated as undocumented; days 0 and 1 are assumed to represent the weekend.
- **Segmentation interpretation:** Cluster labels describe observed purchasing patterns and should not be interpreted as permanent customer characteristics.
- **Dataset dependency:** Results depend on the dataset sample, feature engineering, train-test split, and model configuration.

## Future Scope

- Improve reorder prediction through additional feature engineering and model tuning.
- Explore more efficient approaches for large-scale customer-product prediction.
- Add richer customer-segment comparisons and interactive visualisations.
- Evaluate the models on additional customer samples and alternative validation settings.
- Improve the usability and interpretability of the Streamlit dashboard.

## Academic Information

**Project:** Instacart Customer Segmentation & Reorder Prediction System  
**Course:** Machine Learning Lab  
**Course Code:** 24CS01TP0501  
**Repository:** Machine-Learning-Project

## Dataset Acknowledgement

This project uses the Instacart Online Grocery Shopping Dataset 2017, provided through the Kaggle competition *Instacart Market Basket Analysis*. The dataset is used for educational and machine learning experimentation purposes, subject to the competition's terms of use.
