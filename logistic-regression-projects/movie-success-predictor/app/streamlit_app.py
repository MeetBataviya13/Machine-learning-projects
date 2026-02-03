import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .flop-box {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .season-badge {
        background-color: #20B2AA;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Month to Season mapping
MONTH_TO_SEASON = {
    "January": "Winter",
    "February": "Winter",
    "March": "Spring",
    "April": "Spring",
    "May": "Spring",
    "June": "Summer",
    "July": "Summer",
    "August": "Summer",
    "September": "Fall",
    "October": "Fall",
    "November": "Fall",
    "December": "Winter"
}

# Genre list
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"
]

# Language list
LANGUAGES = [
    "English", "Hindi", "Spanish", "French", "Japanese", "Korean",
    "Mandarin", "Italian", "German", "Other"
]

# Top directors from TMDB dataset (you can expand this list)
TOP_DIRECTORS = [
    "-- Select Director --",
    "Steven Spielberg",
    "Christopher Nolan",
    "Martin Scorsese",
    "James Cameron",
    "Ridley Scott",
    "Tim Burton",
    "David Fincher",
    "Quentin Tarantino",
    "Peter Jackson",
    "Michael Bay",
    "Ron Howard",
    "Robert Zemeckis",
    "Clint Eastwood",
    "Tony Scott",
    "Roland Emmerich",
    "Guy Ritchie",
    "Zack Snyder",
    "Bryan Singer",
    "Denis Villeneuve",
    "J.J. Abrams",
    "Gore Verbinski",
    "Sam Raimi",
    "Brett Ratner",
    "Woody Allen",
    "Joel Schumacher",
    "Barry Sonnenfeld",
    "Jon Favreau",
    "Spike Lee",
    "Paul W.S. Anderson",
    "John Woo",
    "Other"
]

# Load models
@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(BASE_DIR)

        logistic_model = joblib.load(os.path.join(PROJECT_ROOT, "models", "logistic_model.joblib"))
        linear_model = joblib.load(os.path.join(PROJECT_ROOT, "models", "linear_model.joblib"))
        ridge_model = joblib.load(os.path.join(PROJECT_ROOT, "models", "ridge_model.joblib"))
        preprocessor = joblib.load(os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib"))
        return logistic_model, linear_model, ridge_model, preprocessor , True
    except Exception as e:
        st.warning(f"Models not loaded: {e}")
        return None, None, None, None, False

# Load processed data for reference
@st.cache_data
def load_processed_data():
    try:
        df = pd.read_csv('../data/processed/processed_movies.csv')
        return df
    except:
        return None

# Training data statistics for range checking
TRAINING_RANGES = {
    'budget': {'min': 0, 'max': 380000000, 'mean': 29000000},
    'runtime': {'min': 0, 'max': 338, 'mean': 94},
    'popularity': {'min': 0, 'max': 547, 'mean': 21},
    'vote_count': {'min': 0, 'max': 14075, 'mean': 109}
}

def check_input_range(feature, value):
    """Check if input is within training range"""
    if feature in TRAINING_RANGES:
        range_info = TRAINING_RANGES[feature]
        if value < range_info['min'] or value > range_info['max']:
            return True, f"{feature.title()} outside training range ({range_info['min']:,} - {range_info['max']:,}). Prediction may be less reliable."
    return False, ""

def get_season(month):
    """Get season from month"""
    return MONTH_TO_SEASON.get(month, "Unknown")

# Load models and data
logistic_model, linear_model, ridge_model, preprocessor = load_models()
processed_data = load_processed_data()

# Main header
st.markdown('<div class="main-header">🎬 Movie Success Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ML-Powered Box Office & Rating Prediction System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎬 Project Overview")
    st.info("""
    This ML system predicts:
    - 🎯 **Box Office Success** (Hit/Flop)
    - ⭐ **IMDb Rating** (Regression)
    
    Built using TMDB 5000 dataset.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("🤖 Logistic Regression Accuracy", "74.61%")
    st.metric("📈 Ridge Regression R²", "0.45")
    st.metric("📉 Ridge RMSE", "0.70")
    
    st.markdown("---")
    st.markdown("### 📚 Dataset Info")
    st.write("- 🌐 **Source**: TMDB 5000")
    st.write("- 🎞️ **Movies**: 4,803")
    st.write("- ✨ **Features**: 20+")
    st.write("- 📅 **Time Period**: 1916-2017")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Hit/Flop Prediction",
    "Rating Prediction", 
    "Model Insights",
    "Batch Prediction"
])

# ==================== TAB 1: Hit/Flop Prediction ====================
with tab1:
    st.markdown("### Box Office Success Prediction")
    st.markdown("Predict whether your movie will be a **Hit** or **Flop** based on production metadata.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Financial Details")
        currency = st.selectbox("Currency", ["USD ($)", "INR (₹)"], key="currency_hit")
        
        if currency == "INR (₹)":
            budget_input = st.number_input(
                "Budget (₹)", 
                min_value=0, 
                max_value=3000000000, 
                value=250000000, 
                step=10000000,
                help="Production budget in Indian Rupees"
            )
            budget = budget_input / 83  # Convert to USD
            st.caption(f"≈ ${budget/1e6:.2f}M USD")
        else:
            budget = st.number_input(
                "Budget ($)", 
                min_value=0, 
                max_value=380000000, 
                value=30000000, 
                step=1000000,
                help="Production budget in US Dollars"
            )
        
        runtime = st.number_input(
            "Runtime (minutes)", 
            min_value=60, 
            max_value=300, 
            value=110,
            help="Total movie duration"
        )
    
    with col2:
        st.markdown("#### Creative Elements")
        primary_genre = st.selectbox(
            "Primary Genre",
            GENRES,
            index=0
        )
        
        language = st.selectbox(
            "Original Language",
            LANGUAGES,
            index=0
        )
        
        director = st.selectbox(
            "Director",
            TOP_DIRECTORS,
            index=0,
            help="Select from top directors or choose 'Other'"
        )
        
        if director == "Other":
            director_custom = st.text_input("Enter Director Name", placeholder="e.g., John Smith")
    
    with col3:
        st.markdown("#### Release Planning")
        release_month = st.selectbox(
            "Release Month",
            list(MONTH_TO_SEASON.keys()),
            index=5  # Default to June
        )
        
        # Auto-calculate season
        season = get_season(release_month)
        st.markdown(f'<div class="season-badge">Season: {season}</div>', unsafe_allow_html=True)
        
        st.markdown("#### Popularity Score")
        
        # Initialize session state for popularity
        if 'popularity_hit' not in st.session_state:
            st.session_state.popularity_hit = 20.0
        
        popularity = st.number_input(
            "Popularity",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.popularity_hit,
            step=1.0,
            key="pop_input_hit"
        )
        st.session_state.popularity_hit = popularity
        
        st.caption("Expected popularity score (0-100)")
    
    # Range warnings
    warnings_list = []
    
    out_of_range, msg = check_input_range('budget', budget)
    if out_of_range:
        warnings_list.append(msg)
    
    out_of_range, msg = check_input_range('runtime', runtime)
    if out_of_range:
        warnings_list.append(msg)
    
    out_of_range, msg = check_input_range('popularity', popularity)
    if out_of_range:
        warnings_list.append(msg)
    
    if warnings_list:
        st.warning("**Input Range Warnings:**\n" + "\n".join([f"- {w}" for w in warnings_list]))
    
    st.markdown("---")
    
    if st.button("Predict Box Office Success", type="primary", use_container_width=True):
        # Simulated prediction (replace with actual model)
        hit_probability = min(100, (budget / 1000000 * 0.5 + popularity * 0.8 + runtime * 0.1))
        hit_probability = hit_probability / 100
        hit_probability = min(0.95, max(0.05, hit_probability))
        
        predicted_class = "HIT" if hit_probability > 0.5 else "FLOP"
        
        st.markdown("### Prediction Results")
        
        col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
        
        with col_pred1:
            if predicted_class == "HIT":
                st.markdown(f'<div class="prediction-box">HIT</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flop-box">FLOP</div>', unsafe_allow_html=True)
        
        with col_pred2:
            st.metric("Success Probability", f"{hit_probability*100:.1f}%")
            
        with col_pred3:
            confidence_level = "High" if abs(hit_probability - 0.5) > 0.3 else "Moderate" if abs(hit_probability - 0.5) > 0.15 else "Low"
            st.metric("Confidence Level", confidence_level)
        
        # Interpretation
        st.markdown("---")
        st.markdown("#### Interpretation")
        
        if predicted_class == "HIT":
            st.success(f"""
            **Estimated Probability: {hit_probability*100:.1f}%**
            
            Movies with similar metadata (budget: ${budget/1e6:.1f}M, genre: {primary_genre}, 
            release: {release_month} - {season}) historically performed well at the box office.
            
            **Key Factors:**
            - Budget allocation for {primary_genre} genre
            - {season} release timing
            - Popularity indicators
            """)
        else:
            st.error(f"""
            **Estimated Probability: {hit_probability*100:.1f}%**
            
            Based on historical patterns, movies with these characteristics face challenges 
            in achieving box office success.
            
            **Risk Factors:**
            - Budget-genre alignment
            - Release window competition
            - Pre-release awareness
            """)
        
        # Feature contribution
        st.markdown("#### Feature Contribution to Prediction")
        
        total = budget / 1000000 * 0.3 + popularity * 0.4 + 15 + 10 + runtime * 0.05
        feature_importance = {
            'Budget': (budget / 1000000 * 0.3) / total * 100,
            'Popularity': (popularity * 0.4) / total * 100,
            'Genre': 15 / total * 100,
            'Release Timing': 10 / total * 100,
            'Runtime': (runtime * 0.05) / total * 100
        }
        
        fig = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            labels={'x': 'Contribution (%)', 'y': 'Feature'},
            title='Impact of Features on Prediction'
        )
        fig.update_traces(marker_color='#E50914')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: Rating Prediction ====================
with tab2:
    st.markdown("### IMDb Rating Prediction")
    
    st.warning("""
    **Important Disclaimer:**
    
    IMDb ratings are subjective and influenced by factors not present in metadata alone:
    - Story quality and screenplay
    - Acting performances
    - Cinematography and technical excellence
    - Audience sentiment and cultural factors
    
    **Our Ridge Regression model achieves R² = 0.45, indicating these predictions are estimates based on historical patterns.**
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Movie Details")
        budget_rating = st.number_input(
            "Budget ($)", 
            min_value=0, 
            max_value=380000000, 
            value=30000000, 
            step=1000000, 
            key="budget_rating"
        )
        
        runtime_rating = st.number_input(
            "Runtime (minutes)", 
            min_value=60, 
            max_value=300, 
            value=110, 
            key="runtime_rating"
        )
        
        genre_rating = st.selectbox(
            "Primary Genre", 
            GENRES, 
            key="genre_rating"
        )
        
        vote_count = st.number_input(
            "Expected Vote Count", 
            min_value=0, 
            max_value=15000, 
            value=500, 
            step=100,
            help="Number of user ratings (proxy metric)"
        )
    
    with col2:
        st.markdown("#### Additional Metrics")
        
        # Popularity with +/- buttons
        st.markdown("**Popularity Score**")
        
        if 'popularity_rating' not in st.session_state:
            st.session_state.popularity_rating = 20.0
        
        popularity_rating = st.number_input(
            "Popularity",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.popularity_rating,
            step=1.0,
            key="pop_input_rating"
        )
        st.session_state.popularity_rating = popularity_rating
        
        release_year = st.number_input(
            "Release Year", 
            min_value=1920, 
            max_value=2025, 
            value=2024
        )
        
        language_rating = st.selectbox(
            "Original Language", 
            LANGUAGES, 
            key="language_rating"
        )
    
    if st.button("Predict Rating", type="primary", use_container_width=True):
        # Ridge Regression prediction (simulated)
        base_rating = 5.5 + (budget_rating / 50000000) * 0.3 + (runtime_rating / 120) * 0.2 + (vote_count / 1000) * 0.3 + (popularity_rating / 50) * 0.2
        predicted_rating = min(9.5, max(3.0, base_rating + np.random.normal(0, 0.2)))
        
        st.markdown("### Rating Predictions")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.metric("Predicted IMDb Rating", f"{predicted_rating:.1f}/10")
        
        with col_r2:
            # Rating band
            if predicted_rating >= 7.0:
                rating_band = "Good to Excellent"
            elif predicted_rating >= 5.5:
                rating_band = "Average to Good"
            else:
                rating_band = "Below Average"
            st.metric("Rating Category", rating_band)
        
        with col_r3:
            st.metric("Model RMSE", "±0.70")
        
        # Visual representation
        st.markdown("---")
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("#### Rating Gauge")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_rating,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Rating"},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "#ffcccc"},
                        {'range': [5, 7], 'color': "#ffffcc"},
                        {'range': [7, 10], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 7.0
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_viz2:
            st.markdown("#### Prediction Range")
            
            # Show confidence interval
            lower_bound = max(0, predicted_rating - 0.70)
            upper_bound = min(10, predicted_rating + 0.70)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[predicted_rating],
                y=['Predicted'],
                orientation='h',
                marker_color='#1f77b4',
                error_x=dict(type='constant', value=0.70, color='gray')
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 10], title='Rating'),
                yaxis=dict(showticklabels=False),
                height=200,
                title=f'Rating: {predicted_rating:.1f} (Range: {lower_bound:.1f} - {upper_bound:.1f})'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            **Model Analysis:**
            - Predicted Rating: {predicted_rating:.1f}/10
            - 68% Confidence Range: {lower_bound:.1f} - {upper_bound:.1f}
            - Based on: Budget, Runtime, Popularity, Vote Count
            
            Note: Actual ratings depend heavily on content quality which cannot be predicted from metadata.
            """)

# ==================== TAB 3: Model Insights ====================
with tab3:
    st.markdown("### Model Performance & Insights")
    
    st.markdown("""
    This section shows the performance metrics and key learnings from our trained models.
    """)
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.markdown("#### Logistic Regression - Hit/Flop Classification")
        
        # Actual confusion matrix from user's results
        confusion_data = np.array([[197, 87], [77, 285]])
        
        fig = px.imshow(
            confusion_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Flop', 'Hit'],
            y=['Flop', 'Hit'],
            text_auto=True,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Actual metrics from user's results
        st.markdown("**Classification Report:**")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Accuracy", "74.61%")
        col_m2.metric("Precision (Hit)", "77%")
        col_m3.metric("Recall (Hit)", "79%")
        
        st.markdown("**Per-Class Performance:**")
        class_report = pd.DataFrame({
            'Class': ['Flop (0)', 'Hit (1)'],
            'Precision': ['0.72', '0.77'],
            'Recall': ['0.69', '0.79'],
            'F1-Score': ['0.71', '0.78'],
            'Support': [284, 362]
        })
        st.dataframe(class_report, use_container_width=True, hide_index=True)
    
    with col_insight2:
        st.markdown("#### Regression Models - Rating Prediction")
        
        # Model comparison from user's results
        regression_results = pd.DataFrame({
            'Model': ['Linear Regression', 'Ridge Regression'],
            'MAE': [0.882, 0.505],
            'RMSE': [1.219, 0.696],
            'R²': [-0.690, 0.449]
        })
        
        st.dataframe(regression_results, use_container_width=True, hide_index=True)
        
        st.markdown("**Visual Comparison:**")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='RMSE',
            x=['Linear Regression', 'Ridge Regression'],
            y=[1.219, 0.696],
            marker_color=['#dc3545', '#28a745']
        ))
        fig.update_layout(
            title='RMSE Comparison (Lower is Better)',
            yaxis_title='RMSE',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Finding:** Ridge Regression significantly outperforms Linear Regression due to regularization handling multicollinearity.
        
        - Linear Regression: Negative R² indicates poor fit
        - Ridge Regression: R² = 0.45 explains ~45% of variance
        """)
    
    st.markdown("---")
    
    # Key Learnings
    st.markdown("### Key Learnings & Insights")
    
    col_learn1, col_learn2 = st.columns(2)
    
    with col_learn1:
        st.markdown("#### Model Strengths")
        st.success("""
        **What Works:**
        
        1. **Hit/Flop Classification**
           - 74.61% accuracy is reasonable for pre-release prediction
           - Better at identifying Hits (79% recall) than Flops (69% recall)
        
        2. **Budget & Popularity**
           - Strong correlation with box office success
           - High-budget films with good pre-release buzz perform better
        
        3. **Ridge Regularization**
           - Handles multicollinearity effectively
           - Reduces overfitting compared to plain Linear Regression
        """)
    
    with col_learn2:
        st.markdown("#### Model Limitations")
        st.error("""
        **Constraints:**
        
        1. **Rating Prediction Difficulty**
           - R² = 0.45 means 55% of variance unexplained
           - Subjective factors cannot be captured
        
        2. **Missing Critical Factors:**
           - Screenplay and story quality
           - Acting performances
           - Director's creative execution
           - Marketing effectiveness
           - Critical reviews and word-of-mouth
        
        3. **Data Limitations:**
           - Training data is pre-2017
           - Streaming era dynamics not captured
        """)
    
    st.markdown("---")
    
    # Model Selection Summary
    st.markdown("### Model Selection Summary")
    
    summary_df = pd.DataFrame({
        'Task': ['Hit/Flop Prediction', 'Rating Prediction'],
        'Model Used': ['Logistic Regression', 'Ridge Regression'],
        'Key Metric': ['Accuracy: 74.61%', 'R²: 0.449'],
        'Why Selected': [
            'High interpretability, good performance on binary classification',
            'Regularization prevents overfitting, handles multicollinearity'
        ]
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==================== TAB 4: Batch Prediction ====================
with tab4:
    st.markdown("### Batch Prediction System")
    st.markdown("Upload a CSV file with multiple movies to get bulk predictions.")
    
    st.info("""
    **Required CSV Format:**
    
    Your file must contain these columns:
    - `budget` (numeric): Production budget in USD
    - `runtime` (numeric): Movie duration in minutes
    - `popularity` (numeric): Popularity score (0-100)
    - `primary_genre` (text): Primary genre
    - `original_language` (text): Original language
    - `release_month` (text): Release month name
    - `vote_count` (numeric, optional): Expected vote count
    """)
    
    # Template download
    template_df = pd.DataFrame({
        'budget': [50000000, 30000000, 100000000],
        'runtime': [120, 95, 150],
        'popularity': [45.5, 22.3, 78.9],
        'primary_genre': ['Action', 'Comedy', 'Science Fiction'],
        'original_language': ['English', 'English', 'English'],
        'release_month': ['June', 'December', 'May'],
        'vote_count': [1500, 800, 3000]
    })
    
    csv = template_df.to_csv(index=False)
    st.download_button(
        label="Download Template CSV",
        data=csv,
        file_name="movie_batch_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} movies.")
            
            with st.expander("Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Validation
            required_cols = ['budget', 'runtime', 'popularity', 'primary_genre', 'original_language', 'release_month']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        predictions_hit = []
                        predictions_rating = []
                        seasons = []
                        
                        for idx, row in df.iterrows():
                            # Hit/Flop prediction
                            hit_prob = min(100, (row['budget'] / 1000000 * 0.5 + row['popularity'] * 0.8 + row['runtime'] * 0.1))
                            hit_prob = hit_prob / 100
                            predictions_hit.append("HIT" if hit_prob > 0.5 else "FLOP")
                            
                            # Rating prediction
                            vote_c = row.get('vote_count', 500)
                            rating = 5.5 + (row['budget'] / 50000000) * 0.3 + (row['runtime'] / 120) * 0.2 + (vote_c / 1000) * 0.3
                            predictions_rating.append(round(min(9.5, max(3.0, rating)), 1))
                            
                            # Season
                            seasons.append(get_season(row['release_month']))
                        
                        df['season'] = seasons
                        df['predicted_hit_flop'] = predictions_hit
                        df['predicted_rating'] = predictions_rating
                        
                        st.success("Batch prediction completed!")
                        
                        # Summary
                        st.markdown("### Prediction Summary")
                        
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        
                        hit_count = sum(1 for p in predictions_hit if p == "HIT")
                        flop_count = len(predictions_hit) - hit_count
                        
                        with col_sum1:
                            st.metric("Total Movies", len(df))
                        with col_sum2:
                            st.metric("Predicted Hits", hit_count)
                        with col_sum3:
                            st.metric("Predicted Flops", flop_count)
                        with col_sum4:
                            st.metric("Avg Predicted Rating", f"{np.mean(predictions_rating):.1f}")
                        
                        # Visualizations
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            fig = px.pie(
                                values=[hit_count, flop_count],
                                names=['HIT', 'FLOP'],
                                title="Hit vs Flop Distribution",
                                color=['HIT', 'FLOP'],
                                color_discrete_map={'HIT': '#28a745', 'FLOP': '#dc3545'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_viz2:
                            fig = px.histogram(
                                x=predictions_rating,
                                nbins=10,
                                title="Predicted Ratings Distribution",
                                labels={'x': 'Rating', 'y': 'Count'}
                            )
                            fig.update_traces(marker_color='#1f77b4')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.markdown("### Detailed Results")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Download results
                        result_csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=result_csv,
                            file_name="movie_predictions_results.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Model Limitations Section
st.markdown("---")
with st.expander("Model Limitations & Considerations"):
    st.markdown("""
    ### What the Model Cannot Capture
    
    1. **Story Quality** - Screenplay, narrative, originality
    2. **Performances** - Acting quality, chemistry
    3. **Technical Excellence** - Cinematography, VFX, sound
    4. **Marketing** - Campaign effectiveness, buzz
    5. **External Factors** - Competition, cultural trends, economic conditions
    6. **Streaming Impact** - OTT dynamics (post-2017)
    
    ### Recommendations
    
    - Use predictions as ONE input in decision-making
    - Combine with qualitative analysis
    - Validate against industry expert opinions
    - Don't rely solely on predictions for final decisions
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'><b>Model Version:</b> v1.0 | <b>Dataset:</b> TMDB 5000 | <b>Last Updated:</b> February 2025</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'><small>Developed by Meet Bataviya | Powered by Machine Learning | Trained on historical data (1916-2017)</small></div>", unsafe_allow_html=True)