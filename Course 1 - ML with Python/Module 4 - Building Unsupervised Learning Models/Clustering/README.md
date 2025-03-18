# 📊 Machine Learning with Python - Module 4: Building Unsupervised Learning Models - Clustering

## 📖 Overview

In **Module 4**, I explored **unsupervised learning techniques** with a focus on **clustering methods**. These methods do not require labeled data but instead identify **patterns and structures** in datasets based on similarity.

Key topics covered in this module:  
✔ **Clustering techniques** (Partition-based, Density-based, and Hierarchical).  
✔ **K-Means clustering** and methods for determining the best K.  
✔ **DBSCAN and HDBSCAN** for density-based clustering.  
✔ **Hierarchical clustering** using agglomerative and divisive approaches.  
✔ **Real-world applications** of clustering in **customer segmentation, anomaly detection, and pattern recognition**.

This module demonstrates **how clustering improves data analysis, dimensionality reduction, and feature selection**, making machine learning models more efficient and interpretable.

---

## 📌 Topics Covered

### 1️⃣ **What is Clustering?**

**Clustering** is an **unsupervised learning technique** that groups **similar data points** into clusters based on their features and relationships. Unlike classification, clustering does **not require labeled data**, meaning that patterns are found **without predefined categories**.

✔ **Clusters are formed naturally** → Data points that are close together in an **N-dimensional feature space** belong to the same group.  
✔ **Identifies hidden structures** → Useful in **customer segmentation, pattern recognition, and anomaly detection**.  
✔ **Helps simplify large datasets** → Reduces the complexity of datasets by summarizing them into representative clusters.

### **📌 Clustering Uses**

🚀 Clustering is widely used for:  
✔ **Customer segmentation** → Grouping customers based on shopping habits.  
✔ **Image segmentation** → Identifying patterns in medical imaging (e.g., tumor detection).  
✔ **Anomaly detection** → Fraud detection or equipment failure prediction.  
✔ **Feature selection** → Identifying important attributes that define data patterns.  
✔ **Data compression** → Replacing raw data points with cluster representatives to reduce storage requirements.

---

## **2️⃣ Types of Clustering Methods**

### **📌 Partition-Based Clustering (K-Means)**

Partition-based clustering divides the dataset into **K** clusters. Each data point is assigned to the closest centroid.  
✔ **Efficient for large datasets** → K-Means is computationally efficient.  
✔ **Works well for well-separated clusters** → Performs well when data points naturally form groups.

📌 **Limitations**  
✖ Requires pre-defining **K** (number of clusters).  
✖ Assumes clusters are **convex and equal in size**, which may not always be true.

---

### **📌 Density-Based Clustering (DBSCAN, HDBSCAN)**

Density-based clustering finds clusters based on **density regions**, rather than assuming a fixed number of clusters.  
✔ **Works well for irregularly shaped clusters**.  
✔ **Can detect noise and outliers**, unlike K-Means.  
✔ **No need to specify K**.

📌 **Limitations**  
✖ Struggles with clusters of **varying density**.  
✖ Sensitive to **hyperparameter selection** (ε and MinPts).

---

### **📌 Hierarchical Clustering (Agglomerative & Divisive)**

Hierarchical clustering builds a **tree-like structure (dendrogram)** showing how clusters are related.

✔ **Agglomerative Clustering** (Bottom-Up) → Starts with individual points and merges them into clusters.  
✔ **Divisive Clustering** (Top-Down) → Starts with one large cluster and splits it into smaller clusters.

📌 **Limitations**  
✖ **Computationally expensive** for large datasets.  
✖ **Does not work well with noisy data**.

📌 **Distance Metrics for Hierarchical Clustering**  
Hierarchical clustering relies on **distance measures** to determine cluster similarity:  
✔ **Single Linkage** → Merges clusters based on the closest points.  
✔ **Complete Linkage** → Uses the farthest points for cluster merging.  
✔ **Average Linkage** → Uses the mean distance between clusters.

---

## **3️⃣ K-Means Clustering: Centroid-Based Clustering**

✔ **Divides data into K clusters** → Each cluster is represented by a centroid.  
✔ **Minimizes intra-cluster variance** → Points assigned to the nearest centroid.

📌 **Algorithm Steps**  
1️⃣ Select K random centroids.  
2️⃣ Assign each point to the nearest centroid.  
3️⃣ Recalculate centroids based on cluster members.  
4️⃣ Repeat until centroids stop changing (convergence).

📌 **How to Determine K?**  
✔ **Elbow Method** → Plots intra-cluster variance vs. K. The **"elbow point"** indicates the best K value.  
✔ **Silhouette Score** → Measures cluster separation; higher values indicate better clustering.  
✔ **Davies-Bouldin Index** → Lower values indicate better clustering quality.

📌 **Challenges of K-Means**  
✖ **Sensitive to initial centroids** → Different initializations can lead to different results.  
✖ **Assumes spherical clusters** → May fail on irregularly shaped data.

📌 **Distance Metrics Used in K-Means**  
✔ **Euclidean Distance** → Default metric, assumes spherical clusters.  
✔ **Manhattan Distance** → Works better for grid-like data (e.g., city maps).  
✔ **Cosine Distance** → Used for text data and high-dimensional data.

---

## **4️⃣ DBSCAN & HDBSCAN: Density-Based Clustering**

✔ **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)

- Uses **ε (radius)** and **MinPts (minimum points)** to define dense regions.
- **Finds arbitrary-shaped clusters**.
- **Identifies outliers** as noise points.

✔ **HDBSCAN** (Hierarchical DBSCAN)

- Automatically adjusts **density thresholds** to find clusters.
- Works better with **varying densities** than DBSCAN.
- Creates a **hierarchical structure** for clusters.

📌 **Comparison of DBSCAN vs. HDBSCAN**  
| **Feature** | **DBSCAN** | **HDBSCAN** |
|-------------|------------|------------|
| Requires ε & MinPts? | ✅ Yes | ❌ No (automatically adjusts) |
| Handles Noise? | ✅ Yes | ✅ Yes |
| Works for Varying Densities? | ❌ No | ✅ Yes |
| Finds Arbitrary Shapes? | ✅ Yes | ✅ Yes |
| Performance | Faster | More computationally expensive |

🚀 **Use DBSCAN** when clusters have **similar density** and the number of clusters is **unknown**.  
🚀 **Use HDBSCAN** when clusters have **varying density** and require **automatic tuning**.

---

## **📂 Final Thoughts**

✅ **Clustering finds hidden patterns** in data without requiring labels.  
✅ **Different clustering methods** suit different data structures.  
✅ **K-Means is efficient but assumes spherical clusters**.  
✅ **DBSCAN/HDBSCAN are better for noisy, irregularly shaped clusters**.  
✅ **Hierarchical clustering provides interpretable cluster relationships**.  
✅ **Choosing K is critical for K-Means; use Elbow & Silhouette methods**.
