# 🧠 Customer Segmentation Web App

🔗 **Live Demo**: [Visit the App](https://13019370-de69-44ce-b075-8ca500fab23e-00-2v7s0ndn5domp.sisko.replit.dev/)


An interactive web application that performs customer segmentation using KMeans clustering. Designed to help businesses better understand their customers based on demographic and spending behavior.

---

## 🚀 Features

- 🔍 **Individual Customer Prediction**  
  Enter a customer's age, income, spending score, and gender to predict their cluster and receive tailored marketing recommendations.

- 📊 **Interactive Visualizations**  
  - 3D Cluster Plot (Age, Income, Spending)
  - Correlation Heatmap
  - Cluster Profiles
  - Age Distribution by Cluster
  - Income vs Spending Scatter
  - Gender Distribution per Cluster

- 📁 **File Upload Support**  
  Upload your own dataset (`.xlsx`) and get segmented results with auto-generated recommendations and downloadable output.

- 📥 **Excel Download**  
  Processed customer segmentation results are downloadable as an Excel file.

- 🎨 **Professional Design**  
  Custom color palette and responsive layout for a user-friendly experience.

---

## 🧪 Tech Stack

- **Backend**: Python, Flask, Pandas, Scikit-learn
- **Frontend**: HTML, Jinja2, Bootstrap, Plotly, Matplotlib, Seaborn
- **Others**: Excel file support, data preprocessing, interactive forms

---

## 🧰 How It Works

1. **Data Preprocessing**  
   - Features used: Age, Annual Income, Spending Score, Spending-to-Income Ratio  
   - Gender encoded as numeric  
   - Data scaled using `StandardScaler`

2. **Clustering**  
   - Model: KMeans (5 clusters)  
   - Visualization of WCSS (Elbow Method) to determine optimal k  
   - Clusters labeled with actionable business recommendations

3. **User Interaction**  
   - Predict cluster based on manual input or uploaded Excel file  
   - Display plots and cluster-wise statistics dynamically

---

## 📸 Screenshots

<!-- Optional: Add screenshots here -->
![3D Cluster Plot](assets/3d_cluster.png)
![Cluster Heatmap](assets/cluster_heatmap.png)

---

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-app.git
   cd customer-segmentation-app
