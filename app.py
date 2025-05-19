from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import base64
from werkzeug.utils import secure_filename
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Professional color palette (teal and amber)
COLOR_PALETTE = {
    'primary': '#008080',  # Teal
    'secondary': '#FFA500',  # Amber
    'accent': '#CD5C5C',  # Indian Red
    'background': '#F5F5F5',
    'text': '#333333',
    'light': '#FFFFFF',
    'dark': '#1A1A1A',
    'success': '#4CAF50',
    'warning': '#FFC107',
    'info': '#17A2B8'
}

# Load and preprocess data
def load_data():
    dataset = pd.read_excel("Mall_Customers.xlsx")
    dataset['Spending_to_Income_Ratio'] = dataset['Spending Score(1-100)'] / dataset['Annual Income']
    dataset['Gender_Encoded'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
    features = ['Age', 'Annual Income', 'Spending Score(1-100)', 'Spending_to_Income_Ratio']
    X = dataset[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    dataset['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return dataset, kmeans, scaler, features

dataset, kmeans, scaler, features = load_data()

# Recommendation mapping
recommendations = {
0: "Target with luxury items and premium services (High income, high spending)",
1: "Offer budget-friendly options and value deals (Low income, high spending)",
2: "Focus on quality essentials and mid-range products (Medium income, medium spending)",
3: "Provide savings plans and investment options (High income, low spending)",
4: "Suggest trendy, affordable fashion (Young, medium spending)",
5: "Introduce loyalty programs and discounts to retain this cautious spender group (Low income, low spending)",
6: "Highlight travel, lifestyle, and digital services (Young, high spending, mid income)",
7: "Promote premium education and family-focused products (Middle-aged, high income, moderate spending)",
8: "Offer credit products and financial advice (Recently joined customers with high income but low engagement)",
9: "Upsell luxury memberships and VIP experiences (Very high income and spending, frequent buyers)"
}


# Helper function to plot to base64
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

# Generate all visualizations
def generate_visualizations():
    # Elbow Method Plot
    wcss = []
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_temp.fit(scaler.transform(dataset[features]))
        wcss.append(kmeans_temp.inertia_)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o', color=COLOR_PALETTE['primary'])
    plt.title('Elbow Method', fontweight='bold')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    elbow_plot = plot_to_base64(plt.gcf())
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap='YlOrBr', center=0)
    plt.title('Feature Correlation Matrix', fontweight='bold')
    correlation_plot = plot_to_base64(plt.gcf())
    
    # Cluster Profiles Heatmap
    cluster_profiles = dataset.groupby('Cluster')[features].mean()
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_profiles.T, annot=True, cmap='YlOrBr', fmt='.1f')
    plt.title('Average Feature Values by Cluster', fontweight='bold')
    cluster_heatmap = plot_to_base64(plt.gcf())
    
    # Age Distribution by Cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataset, x='Cluster', y='Age', palette='YlOrBr')
    plt.title('Age Distribution by Cluster', fontweight='bold')
    age_dist_plot = plot_to_base64(plt.gcf())
    
    # Income vs Spending Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dataset, x='Annual Income', y='Spending Score(1-100)', 
                   hue='Cluster', palette='YlOrBr', s=100)
    plt.title('Income vs Spending by Cluster', fontweight='bold')
    income_spending_plot = plot_to_base64(plt.gcf())
    
    return {
        'elbow_plot': elbow_plot,
        'correlation_plot': correlation_plot,
        'cluster_heatmap': cluster_heatmap,
        'age_dist_plot': age_dist_plot,
        'income_spending_plot': income_spending_plot,
        'cluster_profiles': cluster_profiles
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'customer_id' in request.form:
            customer_id = int(request.form['customer_id'])
            if customer_id in dataset['Customer ID'].values:
                cluster = dataset.loc[dataset['Customer ID'] == customer_id, 'Cluster'].values[0]
                return render_template('result.html', 
                                    customer_id=customer_id,
                                    cluster=cluster,
                                    recommendation=recommendations.get(cluster, "No specific recommendation"),
                                    show_individual=True,
                                    color_palette=COLOR_PALETTE,
                                    now=datetime.now())
        
        elif 'age' in request.form:
            age = float(request.form['age'])
            income = float(request.form['income'])
            spending = float(request.form['spending'])
            gender = request.form.get('gender', '')
            
            new_data = pd.DataFrame([[age, income, spending, spending/income]], 
                                  columns=['Age', 'Annual Income', 'Spending Score(1-100)', 'Spending_to_Income_Ratio'])
            new_data_scaled = scaler.transform(new_data)
            cluster = kmeans.predict(new_data_scaled)[0]
            
            return render_template('result.html',
                                age=age,
                                income=income,
                                spending=spending,
                                gender=gender,
                                cluster=cluster,
                                recommendation=recommendations.get(cluster, "No specific recommendation"),
                                show_individual=True,
                                color_palette=COLOR_PALETTE,
                                now=datetime.now())
    
    visualizations = generate_visualizations()
    
    # 3D Plot
    fig_3d = px.scatter_3d(dataset, 
                          x='Annual Income', 
                          y='Spending Score(1-100)', 
                          z='Age',
                          color='Cluster',
                          hover_data=['Gender'],
                          title='3D Cluster Visualization',
                          color_continuous_scale='tealrose')
    plot_3d = fig_3d.to_html(full_html=False)
    
    return render_template('index.html',
                         elbow_plot=visualizations['elbow_plot'],
                         correlation_plot=visualizations['correlation_plot'],
                         plot_3d=plot_3d,
                         cluster_heatmap=visualizations['cluster_heatmap'],
                         age_dist_plot=visualizations['age_dist_plot'],
                         income_spending_plot=visualizations['income_spending_plot'],
                         cluster_profiles=visualizations['cluster_profiles'].to_html(classes='table table-striped'),
                         color_palette=COLOR_PALETTE,
                         now=datetime.now())

@app.route('/explore', methods=['GET', 'POST'])
def explore():
    graph_types = {
        '3d_scatter': '3D Scatter Plot',
        'heatmap': 'Correlation Heatmap',
        'cluster_profile': 'Cluster Profiles',
        'age_dist': 'Age Distribution',
        'income_spending': 'Income vs Spending',
        'gender_dist': 'Gender Distribution'
    }
    
    selected_graph = request.form.get('graph_type', '3d_scatter')
    selected_feature = request.form.get('feature', 'Annual Income')
    
    # Generate selected visualization
    if selected_graph == '3d_scatter':
        fig = px.scatter_3d(dataset, 
                           x='Annual Income', 
                           y='Spending Score(1-100)', 
                           z='Age',
                           color='Cluster',
                           hover_data=['Gender'],
                           title='3D Cluster Visualization',
                           color_continuous_scale='tealrose')
        graph_html = fig.to_html(full_html=False)
    elif selected_graph == 'heatmap':
        plt.figure(figsize=(10, 6))
        sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap='YlOrBr', center=0)
        plt.title('Feature Correlation Matrix', fontweight='bold')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graph_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}" class="img-fluid">'
        plt.close()
    elif selected_graph == 'cluster_profile':
        cluster_profiles = dataset.groupby('Cluster')[features].mean()
        plt.figure(figsize=(12, 6))
        sns.heatmap(cluster_profiles.T, annot=True, cmap='YlOrBr', fmt='.1f')
        plt.title('Average Feature Values by Cluster', fontweight='bold')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graph_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}" class="img-fluid">'
        plt.close()
    elif selected_graph == 'age_dist':
        fig = px.box(dataset, x='Cluster', y='Age', color='Cluster', 
                     title='Age Distribution by Cluster',
                     color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']])
        graph_html = fig.to_html(full_html=False)
    elif selected_graph == 'income_spending':
        fig = px.scatter(dataset, x='Annual Income', y='Spending Score(1-100)', 
                        color='Cluster', hover_data=['Age', 'Gender'],
                        title='Income vs Spending by Cluster',
                        color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']])
        graph_html = fig.to_html(full_html=False)
    elif selected_graph == 'gender_dist':
        gender_counts = dataset.groupby(['Cluster', 'Gender']).size().unstack()
        fig = px.bar(gender_counts, barmode='group', 
                    title='Gender Distribution by Cluster',
                    color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']])
        graph_html = fig.to_html(full_html=False)
    
    return render_template('explore.html',
                         graph_types=graph_types,
                         selected_graph=selected_graph,
                         features=features,
                         selected_feature=selected_feature,
                         graph_html=graph_html,
                         color_palette=COLOR_PALETTE,
                         now=datetime.now())

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            new_data = pd.read_excel(filepath)
            new_data['Spending_to_Income_Ratio'] = new_data['Spending Score(1-100)'] / new_data['Annual Income']
            new_data_scaled = scaler.transform(new_data[features])
            new_data['Cluster'] = kmeans.predict(new_data_scaled)
            new_data['Recommendation'] = new_data['Cluster'].map(recommendations)
            
            output_filename = 'processed_' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            new_data.to_excel(output_path, index=False)
            
            return render_template('upload_result.html',
                                tables=[new_data.head(10).to_html(classes='data')],
                                download_file=output_filename)
        
        except Exception as e:
            return f"Error processing file: {str(e)}"

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
