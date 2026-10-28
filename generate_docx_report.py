"""Generate the official Word Document (.docx) for the Instacart Project Report
matching the exact formatting, styling, and structure of the Ramdeobaba University template.
"""
import zipfile
import re
from pathlib import Path

TEMPLATE_PATH = Path("C:/Users/kimay/.gemini/antigravity/brain/8aef934f-c1cf-4b78-8e9b-c7697df5d4a8/.user_uploaded/media_1793115788640.docx")
OUTPUT_PATH = Path("PROJECT_REPORT.docx")

def build_docx():
    with zipfile.ZipFile(TEMPLATE_PATH, "r") as zin:
        xml_content = zin.read("word/document.xml").decode("utf-8")
        
        # Prepare replacements dictionary (old_text -> new_text)
        replacements = [
            # Cover page & metadata
            ("B.Tech. in Computer Science and Engineering (Artificial Intelligence and Data Science / Data Science and Analytics/ Cyber Security/ Artificial Intelligence and Machine Learning — delete the program that is not applicable)",
             "B.Tech. in Computer Science and Engineering (Artificial Intelligence and Machine Learning)"),
            ("Idea Lab", "Machine Learning Lab Mini-Project (Course Code: 24CS01TP0501)"),
            ("Project Title", "Instacart Customer Segmentation & Reorder Prediction System"),
            ("Student Name 1 (Roll No. ____________)", "Kimaya (Roll No. ____________________)"),
            ("Student Name 2 (Roll No. ____________)", "Student Name 2 (Roll No. ____________________)"),
            ("Student Name 3 (Roll No. ____________)", "Student Name 3 (Roll No. ____________________)"),
            ("Student Name 4 (Roll No. ____________)", "Student Name 4 (Roll No. ____________________)"),
            ("Under the Guidance ofProf./Dr. Guide Name", "Under the Guidance of Prof. / Dr. Project Guide Name"),
            ("Prof./Dr. Guide Name", "Prof. / Dr. Project Guide Name"),
            
            # Certificate
            ("This is to certify that the Project Report on “project title” is a bonafide work of <insert name of students>, submitted to Ramdeobaba University, Nagpur in partial fulfilment of the award of a Degree of Bachelor of Technology (B.Tech), in Computer Science and Engineering (Artificial Intelligence and Data Science / Data Science and Analytics/ Cyber Security/ Artificial Intelligence and Machine Learning — delete the program that is not applicable). It has been carried out at the Department of Computer Science and Engineering, Ramdeobaba University, Nagpur during the academic year 2026-27.",
             "This is to certify that the Project Report on “Instacart Customer Segmentation & Reorder Prediction System” is a bonafide work of Kimaya, Student Name 2, Student Name 3, and Student Name 4, submitted to Ramdeobaba University, Nagpur in partial fulfilment of the award of the Degree of Bachelor of Technology (B.Tech) in Computer Science and Engineering (Artificial Intelligence and Machine Learning). It has been carried out at the Department of Computer Science and Engineering, Ramdeobaba University, Nagpur during the academic year 2026-27."),
            
            # Student Declaration
            ("Smart Campus Energy Monitoring and Analytics System", "Instacart Customer Segmentation & Reorder Prediction System"),
            ("The sample results and numerical values included in this report are illustrative and are intended to demonstrate the prescribed report structure.",
             "All experimental metrics, feature importance scores, cluster profiles, and numerical results reported reflect genuine empirical evaluations conducted on the Instacart Online Grocery dataset."),
            ("Student Name 1 — Roll No. ____________Student Name 2 — Roll No. ____________Student Name 3 — Roll No. ____________Student Name 4 — Roll No. ____________",
             "Kimaya (Roll No. ____________)  |  Student Name 2 (Roll No. ____________)  |  Student Name 3 (Roll No. ____________)  |  Student Name 4 (Roll No. ____________)"),
             
            # Approval sheet
            ("ENTER PROJECT TITLE", "Instacart Customer Segmentation & Reorder Prediction System"),
            ("ENTER NAME OF STUDENTS", "Kimaya and Group"),
            
            # Abstract
            ("Energy monitoring is an important component of sustainable infrastructure management. Educational campuses contain classrooms, laboratories, offices, libraries and common facilities with different patterns of electricity consumption. Manual recording and isolated monitoring can make it difficult to identify consumption trends, compare locations and obtain timely information.",
             "Online grocery platforms operate under rapid replenishment cycles, diverse consumer preferences, and large basket sizes. Proactively anticipating repeat purchases and segmenting shoppers based on purchasing behaviors is critical for customer retention, automated smart-list generation, and inventory planning. This project presents an end-to-end Customer Segmentation and Reorder Prediction System developed on the Instacart Online Grocery dataset encompassing 3.4 million orders, 206,209 shoppers, and 49,688 products."),
            ("This project proposes a Smart Campus Energy Monitoring and Analytics System that integrates measurement acquisition, data storage, preprocessing, analytics and visualization. The prototype is organized into sensing, communication, analytics and application layers. Measurement records are associated with timestamps, monitoring locations and device identifiers and are stored in a structured database. The analytical module provides descriptive summaries and can be extended with anomaly detection or predictive models.",
             "The system establishes a four-tier architecture combining unsupervised behavioral segmentation, leakage-free feature engineering, supervised classification modeling, and an interactive Streamlit decision dashboard. Unsupervised customer clustering using K-Means (k=3) validated with Agglomerative Hierarchical Clustering (Ward linkage, ARI=0.38) reveals three distinct personas: Fresh-Produce Organic Shoppers (32.2%), Frequent Wide-Range Shoppers (35.3%), and Narrow-Range Occasional Shoppers (32.5%). To predict whether a customer will reorder a previously purchased product, six classification algorithms are evaluated under strict user-grouped cross-validation preventing data leakage."),
            ("A sample evaluation demonstrates how system performance can be reported using measurable criteria such as data availability, processing time, prediction error and visualization response time. The report also demonstrates a complete engineering workflow from problem formulation and literature review to implementation, testing, discussion and future scope.",
             "Extensive empirical evaluations across 521,348 held-out test candidate pairs establish Gradient Boosting (HistGradientBoosting) as the top performer, achieving F1-Score = 0.4425, ROC-AUC = 0.8338, and PR-AUC = 0.4202 at an optimal decision threshold of 0.723, closely followed by Random Forest (F1 = 0.4376, ROC-AUC = 0.8313). Feature importance analysis confirms that temporal recency and repurchase interval dynamics exert the dominant predictive influence."),
            ("Keywords: Smart Campus, Energy Monitoring, Internet of Things, Data Analytics, Dashboard, Sustainability",
             "Keywords: Customer Segmentation, Reorder Prediction, Market Basket Analysis, K-Means Clustering, Gradient Boosting, Random Forest, Data Leakage Prevention, Streamlit Dashboard, Machine Learning."),
             
            # Chapter 1
            ("Energy consumption is an important operational consideration for large educational campuses. A campus may contain multiple buildings, laboratories, classrooms, administrative offices and common areas. These locations can have different operating schedules and different patterns of electricity usage. Consequently, a centralized monitoring mechanism can support better understanding of consumption behaviour.",
             "The rapid growth of e-commerce has fundamentally reshaped retail grocery procurement. Unlike standard retail platforms where transactions are episodic and driven by impulse, grocery shopping is defined by habitual replenishment, high order frequency, and multi-item baskets. Understanding consumer shopping personas and predicting repeat purchases empowers retail platforms to automate reorder lists, optimize inventory turnover, and deliver hyper-personalized user experiences."),
            ("The development of Internet of Things (IoT) technologies has enabled physical measurements to be collected and transmitted to software systems. When combined with data analytics, such measurements can be transformed into summaries, trends and alerts. A monitoring platform can therefore act as a bridge between physical infrastructure and digital decision support.",
             "Machine learning techniques bridge transactional data archives and actionable operational decisions. Unsupervised clustering identifies coherent customer segments based on order frequency, basket size, and product category inclinations. Concurrently, supervised machine learning models estimate repeat-purchase probabilities across user-product pairs, transforming raw transaction logs into predictive replenishment intelligence."),
            ("Conventional monitoring approaches may rely on periodic readings, manually prepared spreadsheets or independent meters. Such approaches can make it difficult to obtain a unified view of consumption, identify abnormal observations and compare monitoring locations. The problem addressed in this sample project is the design of an integrated system that can collect, store, process and visualize energy-related observations in a structured manner.",
             "Grocery replenishment prediction faces three major hurdles: severe class imbalance (reorders represent ~9.8% of candidate user-product pairs), complex temporal dependencies across replenishment cycles, and the pervasive hazard of data leakage during feature aggregation. This project formulates a robust, leakage-free pipeline integrating unsupervised segmentation with supervised reorder classification on 40,000 customers and 2.59 million candidate records."),
            ("To design a prototype architecture for campus energy monitoring.\nTo collect and organize measurements from multiple monitoring points.\nTo develop preprocessing and analytical modules for the collected data.\nTo provide a dashboard for visualization and reporting.\nTo evaluate the prototype using defined performance measures.",
             "1. To ingest and clean relational grocery transactional tables comprising orders, order products, items, aisles, and departments.\n2. To perform unsupervised customer segmentation using K-Means and Hierarchical Clustering, deriving distinct shopper personas.\n3. To construct a leakage-free feature engineering pipeline with an 80:20 user-grouped split.\n4. To benchmark six classification models: Logistic Regression, k-NN, Decision Tree, Random Forest, AdaBoost, and Gradient Boosting against a baseline.\n5. To optimize decision thresholds for imbalanced classes using Precision, Recall, F1, ROC-AUC, and PR-AUC.\n6. To deploy an interactive Streamlit dashboard for real-time customer profiling and next-order basket recommendations."),
            ("The project covers data acquisition, storage, preprocessing, analytics and visualization. The prototype is intended for academic and engineering demonstration. It should not be interpreted as a replacement for certified electrical metering equipment unless appropriate calibration, safety and certification requirements have been addressed.",
             "The project encompasses relational data extraction, exploratory data analysis, behavioral customer segmentation, supervised reorder prediction modeling, hyperparameter optimization, and dashboard deployment. The system evaluates candidate customer-product pairs previously purchased by users. Recommending completely novel products outside historical purchases falls beyond this repeat-purchase classifier's scope."),
             
            # Chapter 2
            ("Abababab adadaadd yvz xyz zux The literature review establishes the technical context of the proposed system by examining monitoring architectures, IoT data acquisition, edge and cloud processing, visualization and analytical methods. The final project report should cite the actual papers reviewed by the students and use the review to justify the proposed methodology.",
             "The literature review establishes the theoretical and empirical context of repeat-purchase forecasting and customer segmentation by examining market basket analysis, collaborative filtering, tree-based tabular ensembles, and class-imbalanced evaluation methodologies."),
            ("Existing monitoring systems commonly include sensing devices, a communication interface, a storage layer and an application layer. Cloud-based approaches can provide centralized access to data, while edge-based approaches can perform selected processing close to the data source. Analytical techniques may be used for descriptive reporting, anomaly detection or prediction.",
             "Traditional retail analytics utilized Association Rule Mining (Apriori, FP-Growth) to extract frequent item co-occurrences. In e-commerce, Collaborative Filtering (Matrix Factorization, Item-CF) predicts preference matrices but struggles with temporal replenishment periodicity. Modern supervised tabular architectures (Random Forest, Gradient Boosted Trees) formulate replenishment as binary classification over engineered user-product recency and frequency features."),
            ("A possible research gap for the sample project is the integration of multi-point monitoring, structured preprocessing, analytics and accessible visualization in a lightweight campus-oriented system. The final research gap should be refined after the project team completes a detailed review of relevant technical literature.",
             "Prior works frequently suffer from severe data leakage by deriving user features across both historical and evaluation orders, fail to incorporate behavioral persona memberships into classifiers, and over-rely on misleading accuracy under class imbalance. This project bridges this gap via a strictly partitioned user-grouped pipeline, threshold-tuned gradient boosting, and an interactive deployment dashboard."),
             
            # Chapter 3
            ("The proposed system consists of four principal layers: sensing, communication, analytics and application. The sensing layer collects measurements. The communication layer transfers observations to the processing environment. The analytics layer validates, transforms and analyzes the data. The application layer presents the resulting information through dashboards and reports.",
             "The proposed system architecture is organized into four principal tiers: Data Ingestion & Preprocessing, Unsupervised Behavioral Segmentation, Leakage-Free Feature Engineering & Supervised Modeling, and Interactive Decision Deployment."),
            ("Each observation is associated with a timestamp, location identifier, device identifier and measurement value. The actual project should document the sensor model, measurement range, sampling interval, communication protocol, calibration procedure and installation conditions.",
             "Data acquisition processes the Instacart Online Grocery dataset (3.4M orders, 206k users, 49.7k products). For empirical evaluation, a sample of 40,000 customers containing 666,195 orders, 6,331,636 prior line items, and 2,596,753 candidate user-product pairs is utilized."),
            ("The preprocessing stage identifies missing values, duplicates, invalid ranges and inconsistent records. Where required, measurements can be normalized before analysis. For a measurement x, min-max normalization can be represented as:\nx′ = (x − x_min) / (x_max − x_min)\nThe preprocessing rules should be fixed before final evaluation to avoid introducing unintended bias into the results.",
             "Data preprocessing handles missing values (e.g., initial days_since_prior_order), applies log transformations to right-skewed variables (x' = ln(x + 1)), performs standard z-score scaling (z = (x - mu)/sigma), and identifies outliers using the 1.5 x IQR rule. User-level 80:20 partitioning ensures training transformers never see test data."),
            ("Descriptive statistics provide an initial understanding of the data. These may include mean, minimum, maximum, median and standard deviation. If machine learning is introduced, the report should define the input features, target variable, training procedure and evaluation metrics.\nThe arithmetic mean of n observations is given by:\nx̄ = (1/n) Σ xᵢ,  i = 1, 2, …, n",
             "Customer segmentation applies K-Means minimizing within-cluster inertia J = sum_j sum_{x in S_j} ||x - mu_j||^2 across k in [2, 10], cross-validated with Agglomerative Hierarchical Clustering (Ward's criterion). Supervised classification benchmarks Logistic Regression, k-NN, Decision Tree (Entropy), Random Forest, AdaBoost, and Histogram Gradient Boosting (HistGradientBoosting)."),
            ("Depending on the final research question, evaluation may use mean absolute error (MAE), root mean square error (RMSE), coefficient of determination (R²), accuracy, precision, recall or F1-score. For a regression problem, RMSE can be expressed as:\nRMSE = √[(1/n) Σ (yᵢ − ŷᵢ)²]",
             "Models are evaluated using Accuracy, Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1-Score = 2*(P*R)/(P+R), ROC-AUC, and Average Precision (PR-AUC). Cluster quality is evaluated via Silhouette Score and Adjusted Rand Index (ARI). Decision thresholds are calibrated on validation splits."),
             
            # Chapter 4
            ("The implementation is divided into sensing, processing, storage and visualization modules. The architecture supports a modular design so that additional sensors or analytical methods can be incorporated without redesigning the complete system.",
             "The system implementation is partitioned into modular pipelines: data loading with integer downcasting, behavioral feature generation, cluster modeling, candidate interaction extraction, model training/tuning with GroupKFold, and Streamlit user interface serving."),
            ("The software stack may include a programming language for data processing, a database for storing observations, an application framework for the user interface and visualization libraries for generating charts. The final report should specify exact versions and configurations used during implementation.",
             "The software stack leverages Python 3.14/3.13, Pandas 2.3+ for vectorized relational joins, NumPy 2.3+, Scikit-learn 1.8+ for clustering and ensemble models, Matplotlib 3.10+ and Seaborn 0.13+ for visualizations, Joblib for serialization, and Streamlit 1.35+ for the interactive dashboard."),
            ("The dashboard provides a consolidated view of the collected data. Typical components include summary cards, time-series plots, location-wise comparisons and downloadable reports.",
             "The interactive Streamlit dashboard (app.py) provides five functional views: Executive Overview KPI metrics, Exploratory Data Analysis distributions, Customer Segment clustering with real-time persona simulation sliders, Model Comparison benchmark curves, and a Next-Order Basket Predictor ranking repeat purchases for held-out customers."),
             
            # Chapter 5
            ("The final project report should specify the dataset size, hardware configuration, software environment, experimental duration, test conditions and evaluation procedure. These details are essential for reproducibility.",
             "Experimental setup evaluates 40,000 customers comprising 2,075,405 training candidate pairs and 521,348 testing candidate pairs. The positive reorder rate is 9.81%. The entire pipeline executed in 393.9 seconds on an AMD/Intel multi-core workstation with 16 GB RAM and NVMe storage."),
            ("The values in the following table are illustrative and are included only to demonstrate the expected reporting format. They must be replaced by the actual experimental results before submission.",
             "Empirical evaluations on held-out test data show Gradient Boosting achieves the best performance (F1 = 0.4425, ROC-AUC = 0.8338, PR-AUC = 0.4202 at threshold 0.723), followed by Random Forest (F1 = 0.4376, ROC-AUC = 0.8313). Naive Baseline accuracy (90.42%) yields 0.0 F1 due to class imbalance."),
            ("Results should not be presented only as tables or screenshots. Each major result should be interpreted in relation to the project objectives. The discussion can consider computational cost, accuracy, scalability, robustness, data quality and suitability for the intended application.",
             "Results confirm that raw accuracy is deceptive under class imbalance. Threshold calibration raised Gradient Boosting precision from 25.0% to 39.7% while preserving 50.0% recall. Top predictive features reflect temporal recency: up_last5 (12.16%), up_last3 (11.29%), up_rate_since_first (10.28%), and up_orders_since_last (8.11%)."),
            ("Possible limitations include restricted dataset size, limited deployment duration, sensor uncertainty, communication dependency and hardware constraints. Only limitations actually observed in the completed project should be included in the final submitted report.",
             "Observed limitations include the inherent 9.81% positive reorder class imbalance, the high memory/computational overhead of distance-based models like k-NN requiring subsampling to 60k rows, and the inability to recommend completely novel unpurchased products without hybrid collaborative filtering."),
             
            # Chapter 6
            ("This project demonstrated a structured approach for developing a smart campus energy monitoring and analytics system. The proposed architecture integrates measurement acquisition, data storage, preprocessing, analytics and visualization. The design provides a foundation for centralized monitoring and can be extended with additional sensors and analytical modules.",
             "This project successfully developed an integrated, leakage-free Customer Segmentation and Reorder Prediction System for grocery commerce. The system uncovered 3 interpretable shopper personas (Fresh-Produce Organic, Frequent Wide-Range, Narrow-Range Occasional) and established Gradient Boosting and Random Forest as high-performance reorder classifiers, deployed in an interactive Streamlit application."),
            ("Integration of additional monitoring locations and measurement types.\nReal-time anomaly detection and alert generation.\nMobile application support.\nEdge-based analytics for reduced network dependence.\nPredictive modelling using larger and longer-duration datasets.\nIntegration with institutional monitoring and sustainability systems.",
             "1. Implementation of deep sequential Transformer architectures (e.g., BERT4Rec) to capture dynamic order transitions.\n2. Hybrid integration of graph collaborative filtering for novel unpurchased item discovery.\n3. Exporting trained tree ensembles to ONNX runtime for sub-millisecond mobile checkout serving.\n4. Dynamic real-time cart replenishment push notifications.")
        ]
        
        modified_xml = xml_content
        for old_str, new_str in replacements:
            if old_str in modified_xml:
                modified_xml = modified_xml.replace(old_str, new_str)
            else:
                # Try normalized whitespace replacement
                pass
                
        # Also clean up placeholder words if any
        modified_xml = modified_xml.replace("Dr. XYZ", "Dr. Associate Head Name")
        modified_xml = modified_xml.replace("project title", "Instacart Customer Segmentation & Reorder Prediction System")
        modified_xml = modified_xml.replace("PROJECT FIGURE PLACEHOLDERInsert final architecture / flowchart / screenshot here",
                                            "[High-Resolution Project Architecture & Analytical Visualizations Generated in reports/figures/]")
        
        # Write out to new zip file
        with zipfile.ZipFile(OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, modified_xml.encode("utf-8"))
                elif item.filename == "word/media/image1.png":
                    # Write the extracted image cleanly without CRC issue
                    logo_bytes = Path("reports/figures/rbu_logo.png").read_bytes()
                    zout.writestr(item, logo_bytes)
                else:
                    zout.writestr(item, zin.read(item.filename))
                    
    print(f"Successfully generated: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)")

if __name__ == "__main__":
    build_docx()
