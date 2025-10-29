## K-Means Clustering: Wine Dataset Analysis

This repository contains a complete, step-by-step implementation of the **K-Means Clustering** algorithm applied to the well-known **Wine Dataset** from scikit-learn.

This project serves as a practical exercise demonstrating key stages of a clustering workflow:

* **Data Preparation:** Loading and inspecting the dataset.
* **Feature Scaling:** Implementing `StandardScaler` to normalize features for distance-based analysis.
* **Optimal K Determination:** Utilizing the **Elbow Method** to find the most appropriate number of clusters ($K$).
* **Model Training & Prediction:** Fitting the `KMeans` model and assigning cluster labels.
* **Evaluation:** Assessing cluster quality using metrics like the **Silhouette Score**.
* **Visualization:** Creating insightful scatter plots to visualize the resulting clusters across key features.

This exercise provides a clear, documented example of unsupervised learning for data segmentation and pattern recognition.

# 🍷 Unsupervised Learning: Comparative Clustering Analysis on the Wine Dataset

This repository documents an in-depth comparative study of three fundamental clustering algorithms—**K-Means, Hierarchical Clustering (Agglomerative), and DBSCAN**—using the classic **Wine Dataset** from scikit-learn. The project focuses on the end-to-end workflow of unsupervised pattern recognition, emphasizing data preparation, optimal parameter selection, and rigorous model evaluation.

## 📊 Project Scope and Dataset Overview

### About the Dataset

The Wine Dataset comprises **178 samples** and **13 distinct chemical features** (e.g., alcohol, malic_acid, flavanoids, color_intensity), all of which are continuous numerical attributes. The dataset inherently possesses a target column (cultivar type), which is **explicitly excluded** during model fitting to maintain the integrity of the unsupervised learning paradigm. The true labels are used *only* for post-clustering validation and comparative analysis.

---

## 🔬 Technical Workflow & Implementation Tasks

### Part 1: Data Preparation & Preprocessing

This section lays the foundation for reliable clustering by ensuring all features contribute equally to the distance metrics.

1.  **Data Ingestion:** Load the Wine dataset using `sklearn.datasets.load_wine` and construct a `pandas.DataFrame` utilizing only the 13 feature columns.
2.  **Feature Scaling Rationale:** Justify the necessity of applying **Standard Scaling** (`StandardScaler`) to the features prior to clustering, addressing the impact of varying magnitudes on distance-based algorithms.
3.  **Data Transformation:** Apply `StandardScaler` to the feature set.

### Part 2: K-Means Clustering Implementation

The K-Means model is implemented and fine-tuned using established techniques.

1.  **Optimal $K$ Determination:** Employ the **Elbow Method** (Inertia analysis) to systematically determine the optimal number of clusters ($K$) for the scaled data.
2.  **Model Execution:** Fit the `KMeans` model using the determined $K$ value.
3.  **Assignment:** Assign the generated cluster labels back to the primary DataFrame.
4.  **Visualization:** Generate informative scatter plots (e.g., Alcohol vs. Color Intensity), colored by the assigned cluster labels, to visually inspect cluster separation.
5.  **Rigorous Evaluation:**
    * Calculate internal cluster validation metrics: **Inertia** and the **Silhouette Score**.
    * Validate against the ground truth (actual wine types) using a **Contingency Table** (`pd.crosstab`) to assess alignment.

### Part 3: Alternative Clustering Methods

Exploration of non-centroid-based clustering techniques to understand their behavior on the dataset.

#### 🌲 Hierarchical Clustering (Agglomerative)
* Apply `AgglomerativeClustering` with $n\_clusters=3$.
* Perform validation using `pd.crosstab` against the target.
* **(Optional)** Visualize the cluster formation process using a **dendrogram** (Linkage Matrix).

#### 🌊 DBSCAN
* Apply the **Density-Based Spatial Clustering** algorithm.
* Systematically test different combinations of the hyper-parameters: **`eps`** (maximum distance) and **`min_samples`** (minimum points).
* Analyze the designation of **Noise Points** (labeled as $-1$) and their implications for outlier detection.
* Validate results using `pd.crosstab`.

---

## 🏆 Part 4: Comparative Analysis and Recommendation

The final step is synthesizing the results from all three models to draw actionable conclusions.

* **Performance Comparison:** Quantify and compare the performance of K-Means, Hierarchical, and DBSCAN algorithms based on Silhouette Scores and alignment with the true cultivar labels.
* **Behavioral Assessment:** Analyze how each algorithm handles the dataset's underlying structure (e.g., assumption of shape, handling of varying densities).
* **Final Recommendation:** Based on the empirical evidence, provide a final, justified recommendation for the most suitable clustering algorithm for this specific dataset and problem context.
