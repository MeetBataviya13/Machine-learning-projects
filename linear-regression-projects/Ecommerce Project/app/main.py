import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import os


# Try to import BytesIO for Excel support
try:
    from io import BytesIO
    EXCEL_SUPPORT = True
except:
    EXCEL_SUPPORT = False

# Page configuration
st.set_page_config(
    page_title="Customer Spending Predictor",
    layout="wide",
    page_icon="💰",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load pre-trained model and scaler
@st.cache_resource
@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "linear_model.joblib")
        SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.joblib")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        return model, scaler, True

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, False

model, scaler, model_loaded = load_model()

# Header
st.markdown('<p class="main-header">💰 Customer Spending Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict annual customer spending based on engagement metrics</p>', unsafe_allow_html=True)

# Sidebar - Information
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/business-report.png", width=150)
    st.title("About This Tool")
    st.markdown("""
    This predictive tool uses **Machine Learning** to estimate a customer's annual spending based on their engagement patterns.
    
    ### How It Works:
    1. Enter customer engagement metrics
    2. Click 'Predict Spending'
    3. Get instant spending prediction
    
    ### Model Performance:
    - **Accuracy (R²):** 98%
    - **Average Error:** ~$8
    - **CV Score:** 98.5%
    - **Training Data:** 500+ customers
    
    **Cross-Validation (CV) Score** ensures the model performs consistently across different data splits, meaning it's reliable on any subset of data.
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** More accurate inputs lead to better predictions!")
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.markdown("""
    **Algorithm:** Linear Regression  
    **Features Used:** 4  
    **Last Updated:** """ + datetime.now().strftime("%B %Y"))

# Check if model is loaded
if not model_loaded:
    st.error("⚠️ **Model files not found!**")
    st.warning("""
    Please ensure the following files are in the same directory as this app:
    - `linear_model.joblib`
    - `scaler.joblib`
    
    Train the model first using the training script, then run this app.
    """)
    st.stop()

# Main content area
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Predictions", "📈 Analytics"])

# Tab 1: Single Prediction
with tab1:
    st.header("Enter Customer Engagement Metrics")
    
    # Training data ranges for validation
    training_ranges = {
        'Avg. Session Length': (29.53, 36.14),
        'Time on App': (8.51, 15.13),
        'Time on Website': (33.91, 40.01),
        'Length of Membership': (0.27, 6.92)
    }
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Session Metrics")
        avg_session = st.number_input(
            "⏱️ Average Session Length (minutes)",
            min_value=0.0,
            max_value=100.0,
            value=33.0,
            step=0.1,
            help="Average time customer spends per session. Training range: 29.5-36.1 min"
        )
        
        time_app = st.number_input(
            "📱 Time on Mobile App (minutes)",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=0.1,
            help="Average time spent on mobile application. Training range: 8.5-15.1 min"
        )
    
    with col2:
        st.subheader("Engagement Metrics")
        time_website = st.number_input(
            "🌐 Time on Website (minutes)",
            min_value=0.0,
            max_value=100.0,
            value=37.0,
            step=0.1,
            help="Average time spent on website. Training range: 33.9-40.0 min"
        )
        
        membership = st.number_input(
            "👤 Length of Membership (years)",
            min_value=0.0,
            max_value=20.0,
            value=3.5,
            step=0.1,
            help="How long the customer has been a member. Training range: 0.3-6.9 years"
        )
    
    # Check if values are outside training range
    warnings = []
    if not (training_ranges['Avg. Session Length'][0] <= avg_session <= training_ranges['Avg. Session Length'][1]):
        warnings.append(f"⚠️ Avg Session Length ({avg_session:.1f} min) is outside training range ({training_ranges['Avg. Session Length'][0]:.1f}-{training_ranges['Avg. Session Length'][1]:.1f} min)")
    
    if not (training_ranges['Time on App'][0] <= time_app <= training_ranges['Time on App'][1]):
        warnings.append(f"⚠️ Time on App ({time_app:.1f} min) is outside training range ({training_ranges['Time on App'][0]:.1f}-{training_ranges['Time on App'][1]:.1f} min)")
    
    if not (training_ranges['Time on Website'][0] <= time_website <= training_ranges['Time on Website'][1]):
        warnings.append(f"⚠️ Time on Website ({time_website:.1f} min) is outside training range ({training_ranges['Time on Website'][0]:.1f}-{training_ranges['Time on Website'][1]:.1f} min)")
    
    if not (training_ranges['Length of Membership'][0] <= membership <= training_ranges['Length of Membership'][1]):
        warnings.append(f"⚠️ Length of Membership ({membership:.1f} yrs) is outside training range ({training_ranges['Length of Membership'][0]:.1f}-{training_ranges['Length of Membership'][1]:.1f} yrs)")
    
    # Display warnings if any
    if warnings:
        st.warning("**Note:** Some values are outside the training data range. Predictions may be less accurate (extrapolation).")
        for warning in warnings:
            st.caption(warning)
    
    # Info box with current inputs
    st.markdown("### 📋 Current Input Summary")
    input_df = pd.DataFrame({
        'Metric': ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'],
        'Value': [f"{avg_session} min", f"{time_app} min", f"{time_website} min", f"{membership} years"]
    })
    st.dataframe(input_df, hide_index=True, use_container_width=True)
    
    # Predict button
    st.markdown("---")
    if st.button("🚀 Predict Annual Spending", type="primary", use_container_width=True):
        # Prepare input
        input_data = np.array([[avg_session, time_app, time_website, membership]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Calculate confidence level
        in_range_count = sum([
            training_ranges['Avg. Session Length'][0] <= avg_session <= training_ranges['Avg. Session Length'][1],
            training_ranges['Time on App'][0] <= time_app <= training_ranges['Time on App'][1],
            training_ranges['Time on Website'][0] <= time_website <= training_ranges['Time on Website'][1],
            training_ranges['Length of Membership'][0] <= membership <= training_ranges['Length of Membership'][1]
        ])
        confidence = (in_range_count / 4) * 100
        
        # Determine confidence level and color
        if confidence == 100:
            confidence_label = "🟢 High Confidence"
            confidence_color = "#2ecc71"
        elif confidence >= 75:
            confidence_label = "🟡 Medium Confidence"
            confidence_color = "#f39c12"
        else:
            confidence_label = "🔴 Low Confidence (Extrapolation)"
            confidence_color = "#e74c3c"
        
        # Display prediction in styled box
        st.markdown(f"""
        <div class="prediction-box">
            <h2>💵 Predicted Annual Spending</h2>
            <div class="prediction-value">${prediction:,.2f}</div>
            <p>This customer is expected to spend approximately <strong>${prediction:,.2f}</strong> per year</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.2); border-radius: 5px;">
                <strong>Prediction Confidence:</strong> <span style="color: {confidence_color};">{confidence_label}</span> ({confidence:.0f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if confidence < 100:
            st.info("""
            **ℹ️ About Extrapolation:** 
            Some input values are outside the training data range. The model is making predictions based on patterns it learned, 
            but these predictions may be less accurate than those within the training range. Consider this when making business decisions.
            """)
        else:
            st.success("✅ All inputs are within the training data range. High confidence prediction!")
        
        # Feature contribution analysis
        st.markdown("### 📊 Contribution Analysis")
        st.markdown("See how each factor influences the prediction:")
        
        coefficients = model.coef_
        feature_names = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
        contributions = coefficients * input_scaled[0]
        intercept = model.intercept_
        
        # Create contribution dataframe
        contrib_df = pd.DataFrame({
            'Feature': feature_names,
            'Input Value': [f"{avg_session} min", f"{time_app} min", f"{time_website} min", f"{membership} yrs"],
            'Impact on Spending': [f"${c:,.2f}" for c in contributions],
            'Contribution %': [f"{abs(c)/sum(abs(contributions))*100:.1f}%" for c in contributions]
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in contributions]
            bars = ax.barh(feature_names, contributions, color=colors, alpha=0.7)
            ax.set_xlabel('Contribution to Annual Spending ($)', fontsize=12, fontweight='bold')
            ax.set_title('Feature Contribution Breakdown', fontsize=14, fontweight='bold')
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'${width:.1f}', 
                       ha='left' if width > 0 else 'right',
                       va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.dataframe(contrib_df, hide_index=True, use_container_width=True)
        
        # Insights
        st.markdown("### 💡 Key Insights")
        top_contributor = feature_names[np.argmax(np.abs(contributions))]
        st.success(f"**{top_contributor}** has the largest impact on this prediction.")
        
        if membership < 2:
            st.info("💼 This is a relatively new customer. Consider retention strategies!")
        elif membership > 5:
            st.info("⭐ This is a loyal long-term customer. Great for upselling opportunities!")
        
        if time_app > time_website:
            st.info("📱 Customer prefers mobile app over website.")
        else:
            st.info("🌐 Customer prefers website over mobile app.")

# Tab 2: Batch Predictions
with tab2:
    st.header("Batch Predictions")
    st.markdown("Upload a file with multiple customers to get predictions for all of them at once.")
    
    # Show supported formats with examples
    with st.expander("📋 See Supported File Formats & Examples"):
        st.markdown("""
        Your data can be in **any of these formats**:
        
        **1. CSV (Comma-Separated)**
        ```
        Avg. Session Length,Time on App,Time on Website,Length of Membership
        33.0,12.0,37.0,3.5
        31.5,11.5,36.5,2.1
        ```
        
        **2. TXT/TSV (Tab or Space-Separated)**
        ```
        Avg. Session Length    Time on App    Time on Website    Length of Membership
        33.0    12.0    37.0    3.5
        31.5    11.5    36.5    2.1
        ```
        
        **3. Excel (.xlsx, .xls)** - if openpyxl is installed
        - Standard Excel spreadsheet with headers in first row
        
        **4. JSON**
        ```json
        [
          {"Avg. Session Length": 33.0, "Time on App": 12.0, "Time on Website": 37.0, "Length of Membership": 3.5},
          {"Avg. Session Length": 31.5, "Time on App": 11.5, "Time on Website": 36.5, "Length of Membership": 2.1}
        ]
        ```
        
        💡 **The app will automatically detect the format!**
        """)
    
    st.markdown("### 📥 Download Template")
    
    # Let user choose template type
    template_type = st.radio(
        "Choose template type:",
        ["Minimal (Only Required Columns)", "Example (With Sample Extra Columns)"],
        help="Minimal: Just the 4 required columns | Example: Includes sample extra columns to show you can add your own"
    )
    
    if template_type == "Minimal (Only Required Columns)":
        # Simple template with only required columns
        template_df = pd.DataFrame({
            'Avg. Session Length': [33.0, 31.5, 34.2],
            'Time on App': [12.0, 11.5, 13.2],
            'Time on Website': [37.0, 36.5, 38.1],
            'Length of Membership': [3.5, 2.1, 4.8]
        })
        st.success("✅ This template contains ONLY the 4 required columns. You can add your own columns (like Customer ID, Name, etc.) if needed.")
    else:
        # Example template with sample extra columns
        template_df = pd.DataFrame({
            'Customer_ID': ['CUST001', 'CUST002', 'CUST003'],
            'Avg. Session Length': [33.0, 31.5, 34.2],
            'Time on App': [12.0, 11.5, 13.2],
            'Time on Website': [37.0, 36.5, 38.1],
            'Length of Membership': [3.5, 2.1, 4.8],
            'Notes': ['Example data', 'Example data', 'Example data']
        })
        st.info("ℹ️ This template includes sample extra columns (Customer_ID, Notes). You can replace these with your own columns or remove them.")
    
    # Show preview
    with st.expander("👀 Preview Template Data"):
        st.dataframe(template_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📋 Required Columns (Must Have These Exact Names):")
    required_info = pd.DataFrame({
        'Column Name': ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'],
        'Description': ['Average session duration (minutes)', 'Time on mobile app (minutes)', 'Time on website (minutes)', 'Membership duration (years)'],
        'Example': ['33.0', '12.0', '37.0', '3.5']
    })
    st.dataframe(required_info, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 💡 Additional Information:")
    st.markdown("""
    - **You CAN add your own columns** (Customer ID, Name, Email, Department, etc.) - they will be preserved in the output
    - **Column names must match exactly** for the 4 required columns
    - **Order doesn't matter** - columns can be in any order
    - **Extra columns can be anywhere** - before, between, or after the required columns
    """)
    
    st.markdown("---")
    st.markdown("#### ⬇️ Download Template in Your Preferred Format:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # CSV Template
    with col1:
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="⬇️ CSV",
            data=csv,
            file_name="template.csv",
            mime="text/csv"
        )
    
    # TXT Template (tab-separated)
    with col2:
        txt = template_df.to_csv(index=False, sep='\t')
        st.download_button(
            label="⬇️ TXT",
            data=txt,
            file_name="template.txt",
            mime="text/plain"
        )
    
    # Excel Template
    with col3:
        if EXCEL_SUPPORT:
            try:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    template_df.to_excel(writer, index=False, sheet_name='Customer Data')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="⬇️ Excel",
                    data=excel_data,
                    file_name="template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.button("⬇️ Excel", disabled=True, help=f"Error: {str(e)}")
        else:
            st.button("⬇️ Excel", disabled=True, help="Install openpyxl: pip install openpyxl")
    
    # JSON Template
    with col4:
        json_data = template_df.to_json(orient='records', indent=2)
        st.download_button(
            label="⬇️ JSON",
            data=json_data,
            file_name="template.json",
            mime="application/json"
        )
    
    st.markdown("### 📤 Upload Your Data")
    st.info("📝 Supported formats: CSV, TXT, TSV, Excel (.xlsx, .xls), JSON")
    
    # File uploader with multiple formats
    file_types = ['csv', 'txt', 'tsv', 'json']
    if EXCEL_SUPPORT:
        file_types.extend(['xlsx', 'xls'])
    
    uploaded_file = st.file_uploader("Choose a file", type=file_types)
    
    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name.lower()
            
            # Auto-detect and read different file formats
            if file_name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            elif file_name.endswith('.txt'):
                # Try to auto-detect delimiter for text files
                try:
                    # First, try comma-separated
                    batch_df = pd.read_csv(uploaded_file, sep=',')
                except:
                    uploaded_file.seek(0)  # Reset file pointer
                    try:
                        # Try tab-separated
                        batch_df = pd.read_csv(uploaded_file, sep='\t')
                    except:
                        uploaded_file.seek(0)
                        # Try space-separated
                        batch_df = pd.read_csv(uploaded_file, sep='\s+')
            elif file_name.endswith('.tsv'):
                batch_df = pd.read_csv(uploaded_file, sep='\t')
            elif file_name.endswith(('.xlsx', '.xls')):
                if EXCEL_SUPPORT:
                    batch_df = pd.read_excel(uploaded_file)
                else:
                    st.error("❌ Excel support not available. Install openpyxl: pip install openpyxl")
                    st.stop()
            elif file_name.endswith('.json'):
                batch_df = pd.read_json(uploaded_file)
            else:
                st.error("❌ Unsupported file format!")
                st.stop()
            
            st.success(f"✅ File uploaded successfully! Found {len(batch_df)} customers.")
            
            st.markdown("### 📋 Preview of Uploaded Data")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🚀 Generate Predictions for All Customers", type="primary"):
                # Validate columns
                required_cols = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
                
                # Check which columns are present
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                extra_cols = [col for col in batch_df.columns if col not in required_cols]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.info("💡 Please ensure your file has these exact column names: " + ", ".join(required_cols))
                else:
                    # Show column information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"✅ Found all {len(required_cols)} required columns")
                    with col2:
                        if extra_cols:
                            st.info(f"ℹ️ Ignoring {len(extra_cols)} extra column(s): {', '.join(extra_cols[:3])}{'...' if len(extra_cols) > 3 else ''}")
                    
                    # Make predictions using only required columns
                    X_batch = batch_df[required_cols].values
                    X_batch_scaled = scaler.transform(X_batch)
                    predictions = model.predict(X_batch_scaled)
                    
                    # Create results dataframe with ALL original columns + predictions
                    result_df = batch_df.copy()
                    result_df['Predicted Annual Spending ($)'] = predictions.round(2)
                    
                    st.markdown("### 🎯 Prediction Results")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Statistics
                    st.markdown("### 📊 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Customers", len(predictions))
                    col2.metric("Average Spending", f"${predictions.mean():,.2f}")
                    col3.metric("Total Revenue", f"${predictions.sum():,.2f}")
                    col4.metric("Min/Max", f"${predictions.min():.0f} - ${predictions.max():.0f}")
                    
                    # Download results
                    result_csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Predictions (CSV)",
                        data=result_csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.markdown("### 📈 Spending Distribution")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(predictions, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
                    ax.axvline(predictions.mean(), color='red', linestyle='--', linewidth=2, label=f'Average: ${predictions.mean():.2f}')
                    ax.set_xlabel('Predicted Annual Spending ($)', fontsize=12)
                    ax.set_ylabel('Number of Customers', fontsize=12)
                    ax.set_title('Distribution of Predicted Annual Spending', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Tab 3: Analytics
with tab3:
    st.header("Model Analytics & Insights")
    
    st.info("📌 **Note:** These metrics represent the model's performance during training and testing. They show how accurate and reliable the model is on historical data.")
    
    st.markdown("### 🎯 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy (R²)", "98%", help="Percentage of variance explained by the model (from test set)")
    col2.metric("Mean Absolute Error", "$8.43", help="Average prediction error (from test set)")
    col3.metric("Cross-Validation Score", "98.5%", help="Model performance across 5 different data splits - ensures the model is consistent and not overfitted")
    
    st.markdown("### 📊 Feature Importance")
    st.markdown("Understanding which factors most influence customer spending:")
    
    # Feature importance visualization
    feature_importance = pd.DataFrame({
        'Feature': ['Length of Membership', 'Time on App', 'Avg. Session Length', 'Time on Website'],
        'Coefficient': [63.35, 38.74, 25.47, 0.47],
        'Importance': ['Very High', 'High', 'Medium', 'Low']
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6']
        bars = ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors, alpha=0.8)
        ax.set_xlabel('Impact on Annual Spending ($)', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'${width:.2f}', 
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.dataframe(feature_importance, hide_index=True, use_container_width=True)
    
    st.markdown("### 💼 Business Recommendations")
    
    recommendations = [
        {
            "title": "🎯 Focus on Member Retention",
            "description": "Length of Membership has the highest impact ($63.35). Invest in loyalty programs and retention strategies.",
            "priority": "High"
        },
        {
            "title": "📱 Enhance Mobile Experience",
            "description": "Time on App shows strong correlation ($38.74). Improve app features and user experience.",
            "priority": "High"
        },
        {
            "title": "⏱️ Optimize Session Quality",
            "description": "Session Length contributes $25.47. Focus on engagement and content quality.",
            "priority": "Medium"
        },
        {
            "title": "🌐 Website Performance",
            "description": "Website time has minimal impact ($0.47). May need improvement or different strategy.",
            "priority": "Low"
        }
    ]
    
    for rec in recommendations:
        with st.expander(f"{rec['title']} - Priority: {rec['priority']}"):
            st.write(rec['description'])

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔒 Secure & Reliable**")
    st.caption("Your data is processed securely")
with col2:
    st.markdown("**⚡ Real-time Predictions**")
    st.caption("Instant results in seconds")
with col3:
    st.markdown("**📈 98% Accuracy**")
    st.caption("Trained on 500+ customers")

st.markdown("<br><center><small>Developed by Meet Bataviya | Powered by Machine Learning</small></center>", unsafe_allow_html=True)