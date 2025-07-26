import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Skills Analytics Dashboard",
    page_icon="../../CHAIRPERSON 2025/ninetyone_logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stMetric > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">' \
' Skills Analytics & Unemployment Forecasting Dashboard</div>', unsafe_allow_html=True)

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar for file upload and configuration
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success(f"✅ Loaded {len(df)} records")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Load sample data if no file uploaded
    if st.session_state.data is None:
        st.info("📝 Using sample data for demonstration")
        sample_data = {
            'id': range(1, 101),
            'full_name': [f'Person_{i}' for i in range(1, 101)],
            'province': np.random.choice(['Eastern Cape', 'Western Cape', 'Gauteng', 'KwaZulu-Natal', 'Limpopo', 'Free State', 'Northern Cape', 'North West', 'Mpumalanga'], 100),
            'qualifications': np.random.choice(['matric', 'diploma', 'degree', 'masters', 'phd', 'certificate'], 100),
            'skills': np.random.choice(['web', 'data science', 'marketing', 'accounting', 'engineering', 'teaching', 'healthcare', 'agriculture'], 100),
            'work_history': np.random.choice(['none', 'intern', 'junior', 'senior', 'manager', 'executive'], 100),
            'entrepreneurship_interest': np.random.choice(['true', 'false'], 100)
        }
        st.session_state.data = pd.DataFrame(sample_data)

# Main dashboard
if st.session_state.data is not None:
    df = st.session_state.data.copy()
    
    # Data preprocessing
    df['entrepreneurship_interest'] = df['entrepreneurship_interest'].map({'true': True, 'false': False})
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", len(df))
    
    with col2:
        entrepreneur_pct = (df['entrepreneurship_interest'].sum() / len(df)) * 100
        st.metric("Entrepreneurship Interest", f"{entrepreneur_pct:.1f}%")
    
    with col3:
        unique_skills = df['skills'].nunique()
        st.metric("Unique Skills", unique_skills)
    
    with col4:
        provinces = df['province'].nunique()
        st.metric("Provinces Covered", provinces)
    
    # Tabs for different analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([" Overview Analytics", " Skills Analysis", " ML Predictions", " Unemployment Forecasting", " AI Insights"])
    
    with tab1:
        st.header(" Overview Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Province distribution
            province_counts = df['province'].value_counts()
            fig = px.pie(values=province_counts.values, names=province_counts.index, 
                        title="Candidate Distributions by Province")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Qualifications distribution
            qual_counts = df['qualifications'].value_counts()
            fig = px.bar(x=qual_counts.index, y=qual_counts.values, 
                        title="Qualifications Distribution per province")
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills vs Province heatmap
        st.subheader("Skills Distribution Across Provinces, so light colors indicate higher counts, and dark colors indicate lower counts.")
        skills_province = pd.crosstab(df['province'], df['skills'])
        fig = px.imshow(skills_province.values, 
                       x=skills_province.columns, 
                       y=skills_province.index,
                       title="Skills-Province Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header(" Skills Gap Analysis")
        
        # Define critical skills for South Africa
        critical_skills = ['data science', 'web', 'engineering', 'healthcare']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Skills shortage analysis
            skills_counts = df['skills'].value_counts() #what doe this do ?
            shortage_analysis = pd.DataFrame({
                'Skill': skills_counts.index,
                'Current_Count': skills_counts.values,
                'Demand_Score': np.random.randint(70, 100, len(skills_counts)),
                'Shortage_Gap': np.random.randint(20, 80, len(skills_counts))
            })
            
            fig = px.scatter(shortage_analysis, x='Current_Count', y='Demand_Score', 
                           size='Shortage_Gap', color='Skill',
                           title="Skills Shortage Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Critical skills by province
            critical_df = df[df['skills'].isin(critical_skills)]
            critical_counts = critical_df.groupby(['province', 'skills']).size().reset_index(name='count')
            
            fig = px.bar(critical_counts, x='province', y='count', color='skills',
                        title="Critical Skills by Province")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills recommendation engine
        st.subheader("🎯 Skills Recommendation Engine")
        selected_province = st.selectbox("Select Province for Analysis:", df['province'].unique())
        
        province_data = df[df['province'] == selected_province]
        current_skills = province_data['skills'].value_counts()
        
        # Simulate skill recommendations based on market demand
        recommendations = {
            'High Demand Skills': ['data science', 'web development', 'AI/ML'],
            'Medium Demand Skills': ['digital marketing', 'cloud computing', 'cybersecurity'],
            'Emerging Skills': ['blockchain', 'IoT', 'robotics']
        }
        
        for category, skills in recommendations.items():
            st.write(f"**{category}:** {', '.join(skills)}")
    
    with tab3:
        st.header("🚀 Machine Learning Predictions")
        
        # Entrepreneurship prediction model
        st.subheader("Entrepreneurship Interest Prediction")
        
        # Prepare data for ML
        le_province = LabelEncoder()
        le_skills = LabelEncoder()
        le_qual = LabelEncoder()
        le_work = LabelEncoder()
        
        df_ml = df.copy()
        df_ml['province_encoded'] = le_province.fit_transform(df_ml['province'])
        df_ml['skills_encoded'] = le_skills.fit_transform(df_ml['skills'])
        df_ml['qualifications_encoded'] = le_qual.fit_transform(df_ml['qualifications'])
        df_ml['work_history_encoded'] = le_work.fit_transform(df_ml['work_history'])
        
        # Features for prediction
        features = ['province_encoded', 'skills_encoded', 'qualifications_encoded', 'work_history_encoded']
        X = df_ml[features]
        y = df_ml['entrepreneurship_interest']
        
        # Train Random Forest model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predictions and accuracy
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': ['Province', 'Skills', 'Qualifications', 'Work History'],
                'Importance': rf_model.feature_importances_
            })
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        title="Feature Importance for Entrepreneurship Prediction")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(cm, text_auto=True, 
                           title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        # Neural Network for employment prediction
        st.subheader("🧠 Neural Network Employment Prediction")
        
        # Create a simple neural network
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(4,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                           validation_split=0.2, verbose=0)
        
        # Plot training history
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy'))
        fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'))
        fig.update_layout(title='Neural Network Training History', 
                         xaxis_title='Epoch', yaxis_title='Accuracy')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("📈 Unemployment Forecasting Model")
        
        # Simulate unemployment data
        provinces = df['province'].unique()
        
        # Create synthetic unemployment data
        years = list(range(2020, 2031))
        unemployment_data = []
        
        for province in provinces:
            base_rate = np.random.uniform(15, 35)  # Base unemployment rate
            for year in years:
                # Add some trend and randomness
                trend = -0.5 * (year - 2020)  # Slight improvement over time
                noise = np.random.normal(0, 2)
                rate = max(5, base_rate + trend + noise)  # Minimum 5% unemployment
                unemployment_data.append({
                    'Province': province,
                    'Year': year,
                    'Unemployment_Rate': rate,
                    'Population': np.random.randint(1000000, 5000000)
                })
        
        unemployment_df = pd.DataFrame(unemployment_data)
        
        # Interactive forecasting parameters
        st.subheader("📊 Forecasting Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            education_rate = st.slider("Annual Education Rate (%)", 1, 10, 5)
            
        with col2:
            forecast_years = st.slider("Forecast Years", 5, 20, 10)
            
        with col3:
            selected_province_forecast = st.selectbox("Province for Forecast:", provinces)
        
        # Unemployment forecasting model
        st.subheader("🔮 Unemployment Forecast")
        
        # Filter data for selected province
        province_data = unemployment_df[unemployment_df['Province'] == selected_province_forecast]
        
        # Create forecast
        last_year = province_data['Year'].max()
        last_rate = province_data[province_data['Year'] == last_year]['Unemployment_Rate'].iloc[0]
        
        forecast_data = []
        current_rate = last_rate
        
        for year in range(last_year + 1, last_year + forecast_years + 1):
            # Simulate the impact of education on unemployment
            reduction = education_rate * 0.3  # Each 1% education reduces unemployment by 0.3%
            current_rate = max(3, current_rate - reduction + np.random.normal(0, 0.5))
            
            forecast_data.append({
                'Year': year,
                'Unemployment_Rate': current_rate,
                'Type': 'Forecast'
            })
        
        # Combine historical and forecast data
        historical_data = province_data[['Year', 'Unemployment_Rate']].copy()
        historical_data['Type'] = 'Historical'
        
        forecast_df = pd.DataFrame(forecast_data)
        combined_data = pd.concat([historical_data, forecast_df], ignore_index=True)
        
        # Plot forecast
        fig = px.line(combined_data, x='Year', y='Unemployment_Rate', 
                     color='Type', title=f'Unemployment Forecast for {selected_province_forecast}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact analysis
        st.subheader("💡 Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate total impact
            initial_unemployed = (last_rate / 100) * 1000000  # Assume 1M population
            final_unemployed = (current_rate / 100) * 1000000
            jobs_created = initial_unemployed - final_unemployed
            
            st.metric("Jobs Created", f"{jobs_created:,.0f}")
            st.metric("Unemployment Reduction", f"{last_rate - current_rate:.1f}%")
        
        with col2:
            # Economic impact
            gdp_impact = jobs_created * 50000  # Assume R50k per job GDP contribution
            st.metric("GDP Impact", f"R{gdp_impact:,.0f}")
            st.metric("ROI on Education", f"{(gdp_impact / (education_rate * 1000000)):.1f}x")
    
    with tab5:
        st.header("🤖 AI-Powered Insights")
        
        # Skills clustering analysis
        st.subheader("🎯 Skills Clustering Analysis")
        
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for clustering
        cluster_data = df.groupby(['province', 'skills']).size().reset_index(name='count')
        pivot_data = cluster_data.pivot(index='province', columns='skills', values='count').fillna(0)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivot_data)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels
        pivot_data['Cluster'] = clusters
        
        # Visualize clusters
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        cluster_df = pd.DataFrame({
            'Province': pivot_data.index,
            'PC1': pca_data[:, 0],
            'PC2': pca_data[:, 1],
            'Cluster': clusters
        })
        
        fig = px.scatter(cluster_df, x='PC1', y='PC2', color='Cluster', 
                        text='Province', title='Province Skills Clusters')
        st.plotly_chart(fig, use_container_width=True)
        
        # AI-generated recommendations
        st.subheader("🔍 AI-Generated Recommendations")
        
        recommendations = {
            "Skills Development Priority": [
                "Focus on data science and web development skills",
                "Strengthen healthcare and engineering capabilities",
                "Develop digital literacy programs"
            ],
            "Regional Strategies": [
                "Gauteng: Tech hub development",
                "Western Cape: Creative industries",
                "Eastern Cape: Agricultural innovation"
            ],
            "Policy Recommendations": [
                "Increase funding for vocational training",
                "Establish public-private partnerships",
                "Create entrepreneur support programs"
            ]
        }
        
        for category, items in recommendations.items():
            st.write(f"**{category}:**")
            for item in items:
                st.write(f"• {item}")
        
        # Sentiment analysis simulation
        st.subheader("📝 Employment Sentiment Analysis")
        
        # Simulate sentiment data
        sentiments = ['Positive', 'Neutral', 'Negative']
        sentiment_data = {
            'Province': np.random.choice(provinces, 100),
            'Sentiment': np.random.choice(sentiments, 100, p=[0.4, 0.4, 0.2]),
            'Confidence': np.random.uniform(0.6, 0.95, 100)
        }
        
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_summary = sentiment_df.groupby(['Province', 'Sentiment']).size().reset_index(name='Count')
        
        fig = px.bar(sentiment_summary, x='Province', y='Count', color='Sentiment',
                    title='Employment Sentiment by Province')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    st.markdown("""
    ### Expected CSV Format:
    - `id`: Unique identifier
    - `full_name`: Full name of the person
    - `province`: Province location
    - `qualifications`: Educational qualifications
    - `skills`: Primary skills
    - `work_history`: Work experience level
    - `entrepreneurship_interest`: true/false
    """)

# Footer
st.markdown("---")
st.markdown("🔬 **Skills Analytics Dashboard** - Powered by AI & Data Science")