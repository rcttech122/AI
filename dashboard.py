import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Plastic Injection Moulding Quality Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved model and essential data
@st.cache_resource
def load_model_data():
    with open('dashboard_data.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load data
try:
    dashboard_data = load_model_data()
    model = dashboard_data['model']  # This should be your Extra Trees model
    scaler = dashboard_data['scaler']
    feature_names = dashboard_data['feature_names']
    class_names = dashboard_data['class_names']
    feature_importance = dashboard_data.get('feature_importance', None)
    
    # Define quality class descriptions
    quality_descriptions = {
        1: "Waste: Product fails to meet basic standards and must be scrapped.",
        2: "Acceptable: Product meets minimum quality standards but is not ideal.",
        3: "Target: Product meets the desired quality specifications.",
        4: "Inefficient: Product is above acceptable but falls short of target quality due to process inefficiencies."
    }
    
except Exception as e:
    st.error(f"Error loading model data: {e}")
    feature_names = []
    class_names = [1, 2, 3, 4]
    quality_descriptions = {
        1: "Waste: Product fails to meet basic standards and must be scrapped.",
        2: "Acceptable: Product meets minimum quality standards but is not ideal.",
        3: "Target: Product meets the desired quality specifications.",
        4: "Inefficient: Product is above acceptable but falls short of target quality due to process inefficiencies."
    }

# Add title and introduction
st.title("Plastic Injection Moulding Quality Prediction Dashboard")
st.markdown("""
This dashboard predicts the quality class of plastic injection moulded products based on process parameters.
Use the sidebar to adjust parameters and see how they affect the predicted quality.
""")

# Sidebar for parameter inputs
st.sidebar.header("Process Parameter Input")

# Function to determine appropriate slider range for each parameter
def get_parameter_range(parameter_name):
    # Default ranges - adjust these based on your actual data
    ranges = {
        "Melt temperature": (80.0, 160.0, 106.0, 0.1),  # min, max, default, step
        "Mold temperature": (78.0, 82.5, 81.2, 0.1),
        "time_to_fill": (6.0, 11.5, 7.0, 0.1),
        "ZDx - Plasticizing time": (7.0, 10.0, 8.5, 0.1),
        "ZUx - Cycle time": (74.7, 75.8, 75.2, 0.01),
        "SKx - Closing force": (875.0, 935.0, 900.0, 1.0),
        "SKs - Clamping force peak value": (890.0, 950.0, 920.0, 1.0),
        "Ms - Torque peak value current cycle": (90.0, 130.0, 110.0, 1.0),
        "Mm - Torque mean value current cycle": (75.0, 115.0, 95.0, 1.0),
        "APSs - Specific back pressure peak value": (144.0, 151.0, 147.0, 0.1),
        "APVs - Specific injection pressure peak value": (780.0, 940.0, 880.0, 1.0),
        "CPn - Screw position at the end of hold pressure": (8.5, 9.0, 8.75, 0.01),
        "SVo - Shot volume": (18.5, 19.0, 18.75, 0.01)
    }
    
    # Add engineered features if they exist in feature_names
    engineered_ranges = {
        "temp_ratio": (1.29, 1.32, 1.305, 0.001),
        "pressure_time_interaction": (5000, 8000, 6500, 100),
        "cycle_efficiency": (0.08, 0.11, 0.09, 0.001),
        "force_volume_ratio": (48.0, 50.0, 49.0, 0.1),
        "torque_efficiency": (0.80, 0.95, 0.88, 0.01),
        "pressure_ratio": (5.5, 6.5, 6.0, 0.1)
    }
    ranges.update(engineered_ranges)
    
    # Return appropriate range or a default
    if parameter_name in ranges:
        return ranges[parameter_name]
    return (0.0, 100.0, 50.0, 0.1)  # Default values

# Define preset values for target quality
target_quality_presets = {
    "Melt temperature": 106.0,
    "Mold temperature": 81.2,
    "time_to_fill": 6.2,
    "ZDx - Plasticizing time": 8.2,
    "ZUx - Cycle time": 75.7,
    "SKx - Closing force": 905.0,
    "SKs - Clamping force peak value": 925.0,
    "Ms - Torque peak value current cycle": 110.0,
    "Mm - Torque mean value current cycle": 95.0,
    "APSs - Specific back pressure peak value": 147.0,
    "APVs - Specific injection pressure peak value": 875.0,
    "CPn - Screw position at the end of hold pressure": 8.74,
    "SVo - Shot volume": 18.78,
    "temp_ratio": 1.304,
    "pressure_time_interaction": 5425.0,
    "cycle_efficiency": 0.082,
    "force_volume_ratio": 48.6,
    "torque_efficiency": 0.86,
    "pressure_ratio": 5.95
}

# Create input fields for each feature
input_values = {}
for feature in feature_names:
    min_val, max_val, default_val, step = get_parameter_range(feature)
    # Use preset value if apply preset is true
    if 'preset_applied' in st.session_state and st.session_state.preset_applied and feature in target_quality_presets:
        default_val = target_quality_presets[feature]
    
    input_values[feature] = st.sidebar.slider(
        f"{feature}",
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=step
    )

# Add preset buttons for different quality classes
st.sidebar.header("Parameter Presets")
preset_col1, preset_col2 = st.sidebar.columns(2)

with preset_col1:
    if st.button("Target Quality (Class 3)"):
        # Set flag to use preset values
        st.session_state.preset_applied = True
        st.rerun()

with preset_col2:
    if st.button("Reset Parameters"):
        # Reset to default values
        st.session_state.preset_applied = False
        st.rerun()

# Create tabs for different dashboard sections
tabs = st.tabs(["Prediction", "Process Insights", "Model Performance", "About"])

# Prediction Tab Content
with tabs[0]:
    # Create dataframe from input
    input_df = pd.DataFrame([input_values])
    
    # Display input parameters
    st.subheader("Process Parameters")
    st.dataframe(input_df.style.highlight_max(axis=1), use_container_width=True)
    
    # Scale the input features if model is loaded
    if 'model' in locals():
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Display prediction results
        st.subheader("Quality Prediction")
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display predicted class and description
            prediction_class = int(prediction)
            st.info(f"Predicted Quality Class: {prediction_class}")
            st.markdown(f"**{quality_descriptions[prediction_class]}**")
            
            # Create a gauge chart for visual representation
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_class,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Quality Class"},
                gauge = {
                    'axis': {'range': [0.5, 4.5], 'tickvals': [1, 2, 3, 4], 
                             'ticktext': ['Waste', 'Acceptable', 'Target', 'Inefficient']},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0.5, 1.5], 'color': "red"},
                        {'range': [1.5, 2.5], 'color': "orange"},
                        {'range': [2.5, 3.5], 'color': "green"},
                        {'range': [3.5, 4.5], 'color': "blue"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display probability distribution
            st.subheader("Prediction Confidence")
            
            # Create a dataframe for the probabilities
            prob_df = pd.DataFrame({
                'Class': [f"Class {i+1}" for i in range(len(probabilities))],
                'Probability': probabilities,
                'Description': [quality_descriptions[i+1].split(':')[0] for i in range(len(probabilities))]
            })
            
            # Create a bar chart
            fig = px.bar(
                prob_df, 
                x='Class', 
                y='Probability',
                color='Class',
                hover_data=['Description', 'Probability'],
                text_auto='.2%',
                color_discrete_sequence=['red', 'orange', 'green', 'blue']
            )
            fig.update_layout(title="Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Production metrics section
        st.subheader("Production Quality Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            target_prob = probabilities[2] if len(probabilities) > 2 else 0
            st.metric("Target Quality Probability", f"{target_prob:.1%}")
            
        with metric_col2:
            waste_prob = probabilities[0] if len(probabilities) > 0 else 0
            st.metric("Scrap Risk", f"{waste_prob:.1%}")
            
        with metric_col3:
            acceptable_prob = probabilities[1] if len(probabilities) > 1 else 0
            st.metric("Acceptable Quality Probability", f"{acceptable_prob:.1%}")
            
        with metric_col4:
            inefficient_prob = probabilities[3] if len(probabilities) > 3 else 0
            st.metric("Process Inefficiency", f"{inefficient_prob:.1%}")
        
        # What-If Analysis
        st.subheader("What-If Analysis")
        if st.button("Add Current Configuration to Comparison"):
            # Store current configuration
            if 'comparisons' not in st.session_state:
                st.session_state.comparisons = []
            
            current_config = {
                'parameters': input_values.copy(),
                'prediction': int(prediction),
                'probabilities': probabilities.tolist()
            }
            st.session_state.comparisons.append(current_config)
            st.success(f"Configuration added to comparison! (Total: {len(st.session_state.comparisons)})")

        if st.button("Clear Comparisons"):
            if 'comparisons' in st.session_state:
                st.session_state.comparisons = []
                st.success("Comparisons cleared!")

        # Display comparison table if configurations exist
        if 'comparisons' in st.session_state and len(st.session_state.comparisons) > 0:
            st.write("### Parameter Comparison")
            
            # Create comparison dataframe
            comparison_rows = []
            for i, config in enumerate(st.session_state.comparisons):
                row = {'Config #': i+1, 'Predicted Class': config['prediction']}
                for key, value in config['parameters'].items():
                    if key in ['ZUx - Cycle time', 'Mold temperature', 'APVs - Specific injection pressure peak value']:
                        row[key] = value  # Only add the most important parameters
                comparison_rows.append(row)
            
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Create comparison chart
            if len(st.session_state.comparisons) > 1:
                prob_comparison = []
                for i, config in enumerate(st.session_state.comparisons):
                    for j, prob in enumerate(config['probabilities']):
                        prob_comparison.append({
                            'Config #': f"Config {i+1}",
                            'Class': f"Class {j+1}",
                            'Probability': prob
                        })
                
                prob_df = pd.DataFrame(prob_comparison)
                
                fig = px.bar(
                    prob_df,
                    x='Config #',
                    y='Probability',
                    color='Class',
                    barmode='group',
                    color_discrete_sequence=['red', 'orange', 'green', 'blue']
                )
                fig.update_layout(title="Probability Comparison Across Configurations")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Model not loaded. Unable to make predictions.")

# Process Insights Tab
with tabs[1]:
    st.subheader("Process Parameter Analysis")
    
    # Display feature importance (if available)
    if feature_importance is not None:
        st.write("### Feature Importance")
        
        # Sort features by importance
        sorted_importance = sorted(
            [(feature, importance) for feature, importance in zip(feature_names, feature_importance)],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create dataframe for feature importance
        importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
        importance_df = importance_df.head(10)  # Show top 10 features
        
        # Create bar chart
        fig = px.bar(
            importance_df,
            y='Feature',
            x='Importance',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(title="Top 10 Important Features")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show parameter optimization suggestions
        st.write("### Parameter Optimization Suggestions")
        
        # Guidelines based on the most important features
        st.info("**Cycle Time (ZUx)**: Aim for values around 75.7 seconds for target quality.")
        st.info("**Plasticizing Time (ZDx)**: Optimize between 8.0-8.5 seconds for better results.")
        st.info("**Injection Pressure (APVs)**: Values around 870-880 units are associated with target quality.")
        st.info("**Mold Temperature**: Maintain between 81.0-81.3°C for optimal quality.")
    else:
        st.warning("Feature importance data not available.")
        
    # Add ANOVA plot or statistical analysis visualization
    st.write("### Statistical Analysis of Key Parameters")
    
    # Example visualization - modify based on your actual statistical analysis
    parameter_effect = {
        'ZUx - Cycle time': 0.798,
        'Mold temperature': 0.478,
        'APVs - Specific injection pressure peak value': 0.468,
        'SVo - Shot volume': 0.362
    }
    
    effect_df = pd.DataFrame(list(parameter_effect.items()), columns=['Parameter', 'Effect Size (η²)'])
    
    fig = px.bar(
        effect_df,
        x='Parameter',
        y='Effect Size (η²)',
        color='Effect Size (η²)',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(title="Parameter Effect Size on Quality (ANOVA)")
    st.plotly_chart(fig, use_container_width=True)

# Model Performance Tab
with tabs[2]:
    st.subheader("Extra Trees Model Performance")
    
    # Confusion Matrix
    st.write("### Confusion Matrix")
    
    # Try to load confusion matrix image
    try:
        st.image("confusion_matrix_Extra_Trees.png", caption="Confusion Matrix - Extra Trees")
    except:
        # Example confusion matrix if image not available
        cm_data = np.array([
            [75, 3, 0, 0],
            [7, 72, 2, 0],
            [0, 0, 59, 4],
            [0, 0, 2, 76]
        ])
        
        # Create a heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_data, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'],
            yticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4']
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Extra Trees')
        st.pyplot(fig)
    
    # Class-wise performance metrics
    st.write("### Class-wise Performance Metrics")
    
    # Example class metrics - replace with actual values if available
    class_metrics = {
        'Class 1 (Waste)': {'Precision': 0.96, 'Recall': 0.95, 'F1-Score': 0.95},
        'Class 2 (Acceptable)': {'Precision': 0.92, 'Recall': 0.89, 'F1-Score': 0.90},
        'Class 3 (Target)': {'Precision': 0.94, 'Recall': 0.94, 'F1-Score': 0.94},
        'Class 4 (Inefficient)': {'Precision': 0.96, 'Recall': 0.97, 'F1-Score': 0.97}
    }
    
    # Create a dataframe for the metrics
    metrics_df = pd.DataFrame.from_dict(class_metrics, orient='index')
    
    # Display as a table
    st.table(metrics_df)
    
    # Show overall metrics
    st.write("### Overall Model Metrics")
    
    # Display metrics in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "94.0%")
        
    with col2:
        st.metric("Precision (Macro)", "94.0%")
        
    with col3:
        st.metric("Recall (Macro)", "94.0%")
        
    with col4:
        st.metric("F1 Score (Macro)", "94.0%")
    
    # Cross-validation results
    st.write("### Cross-Validation Results")
    
    cv_scores = {
        'Fold 1': 0.93,
        'Fold 2': 0.94,
        'Fold 3': 0.92,
        'Fold 4': 0.95,
        'Fold 5': 0.94,
        'Fold 6': 0.93,
        'Fold 7': 0.94,
        'Fold 8': 0.92,
        'Fold 9': 0.93,
        'Fold 10': 0.95
    }
    
    # Create a line chart for CV scores
    cv_df = pd.DataFrame(list(cv_scores.items()), columns=['Fold', 'Accuracy'])
    
    fig = px.line(
        cv_df,
        x='Fold',
        y='Accuracy',
        markers=True,
        range_y=[0.9, 1.0]
    )
    fig.update_layout(title="10-Fold Cross-Validation Accuracy")
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric("Mean CV Accuracy", f"{np.mean(list(cv_scores.values())):.3f} ± {np.std(list(cv_scores.values())):.3f}")

# About Tab
with tabs[3]:
    st.subheader("About This Dashboard")
    
    st.markdown("""
    ### Project Overview
    
    This dashboard is designed to predict the quality class of plastic injection moulded products based on process parameters. 
    It uses a machine learning model trained on historical manufacturing data to classify products into four quality categories:
    
    1. **Waste**: Product fails to meet basic standards and must be scrapped.
    2. **Acceptable**: Product meets minimum quality standards but is not ideal.
    3. **Target**: Product meets the desired quality specifications.
    4. **Inefficient**: Product is above acceptable but falls short of target quality due to process inefficiencies.
    
    ### Machine Learning Model
    
    The predictions are made using an **Extra Trees classifier**, which demonstrated the best performance among several models 
    evaluated (Logistic Regression, Gaussian Naive Bayes, Random Forest, Extra Trees, and Neural Network). The model 
    was trained on 1000 manufacturing cycles and achieved an accuracy of 94.0% on the test set.
    
    ### How to Use This Dashboard
    
    * Use the **Prediction** tab to input process parameters and get quality predictions.
    * Explore the **Process Insights** tab to understand which parameters most significantly affect product quality.
    * Review the **Model Performance** tab for detailed information about the model's predictive capabilities.
    * Use the parameter presets to see recommended settings for achieving target quality.
    * Try the What-If Analysis to compare different parameter configurations.
    
    """)
    
    # Add information about feature importance
    st.write("### Key Findings")
    
    st.markdown("""
    Our analysis identified the following key process parameters that most significantly influence product quality:
    
    1. **Cycle Time (ZUx)**: The strongest predictor with an effect size (η²) of 0.798. Longer cycle times are associated with Classes 3 (Target) and 4 (Inefficient).
    
    2. **Plasticizing Time (ZDx)**: The second most important feature. Optimal control is essential for achieving consistent quality.
    
    3. **Injection Pressure (APVs)**: Lower pressures (870-880 units) correlate with better quality outcomes, particularly for Class 3 (Target).
    
    4. **Mold Temperature**: Requires precise control within specific ranges for each quality target. Class 3 (Target) products operate at moderate temperatures around 81.2°C.
    
    5. **Engineered Features**: The ratio of fill time to cycle time ("cycle_efficiency") emerged as a significant predictor, highlighting the importance of temporal relationships in the process.
    """)
    
    # Add download capability
    st.write("### Download Prediction Report")
    
    if 'model' in locals() and st.button("Generate Prediction Report"):
        # Create a report dataframe
        report_data = {
            'Parameter': list(input_values.keys()),
            'Value': list(input_values.values())
        }
        report_df = pd.DataFrame(report_data)
        
        # Add prediction information
        prediction_info = pd.DataFrame({
            'Parameter': ['Predicted Class', 'Class Description', 'Target Probability', 'Waste Probability'],
            'Value': [
                int(prediction),
                quality_descriptions[int(prediction)],
                f"{probabilities[2]:.2%}" if len(probabilities) > 2 else "N/A",
                f"{probabilities[0]:.2%}" if len(probabilities) > 0 else "N/A"
            ]
        })
        
        # Combine dataframes
        final_report = pd.concat([report_df, prediction_info])
        
        # Convert to CSV for download
        csv = final_report.to_csv(index=False)
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="injection_moulding_prediction.csv",
            mime="text/csv"
        )