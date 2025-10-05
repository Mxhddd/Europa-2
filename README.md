# Europa-2

# Habitable Worlds Prototype

## Advanced AI Platform for Exoplanet Detection and Habitability Analysis

### Project Overview

The Habitable Worlds Prototype is a comprehensive machine learning platform designed for automated exoplanet validation and Earth similarity assessment. This system leverages ensemble artificial intelligence to analyze Kepler mission data and identify potentially habitable planetary candidates with high accuracy.

### Core Team
- Mohammad
- Waqas 
- Muzammil
- Asadullah

### Key Features

#### AI-Powered Detection Engine
- Ensemble machine learning with algorithms (Random Forest, XGBoost, (classifiers))
- Automated model selection and performance optimization
- Real-time prediction with confidence intervals and uncertainty quantification
- SHAP explainable AI for model interpretability

#### Advanced Habitability Assessment
- Earth Similarity Index (ESI) calculation
- Multi-factor analysis including radius, temperature, flux, and gravity similarity
- Research priority scoring system
- Habitable zone identification and analysis

#### Data Visualization and Analysis
- Interactive 3D exoplanet system visualization
- Real-time parameter filtering and adjustment
- Statistical analysis and correlation matrices
- Comprehensive data export capabilities

### Installation Requirements

#### Python Dependencies
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.13.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.41.0
joblib>=1.2.0

# Installation Steps

## Clone the repository
git clone https://github.com/your-organization/habitable-worlds-prototype.git

## Navigate to project directory
cd habitable-worlds-prototype

## Install required packages
pip install -r requirements.txt

## Launch the application
streamlit run ML.py

# Data Requirements

## Input Format

The application requires CSV files with Kepler mission data containing the following mandatory columns:

kepid: Kepler ID
koi_disposition: Planetary candidate status
koi_period: Orbital period in days
koi_prad: Planetary radius in Earth radii
koi_teq: Equilibrium temperature in Kelvin
Optional Columns for Enhanced Analysis

koi_insol: Insolation flux
koi_model_snr: Signal-to-noise ratio
koi_steff: Stellar effective temperature
koi_srad: Stellar radius
koi_slogg: Stellar surface gravity
Usage Instructions

1. Data Upload

Use the sidebar to upload Kepler data CSV files
Ensure required columns are present in the dataset
The system will automatically process and validate the data

2. Parameter Configuration

Adjust orbital period, temperature, and radius filters
Set Earth Similarity Index thresholds
Configure gravity range and habitable zone preferences

3. Model Training and Analysis

The system automatically trains ensemble models when sufficient labeled data is available
Monitor model performance metrics in the Research Dashboard
Review feature importance and classification reports

4. Candidate Exploration

Examine top Earth similarity candidates in the Candidate Explorer
Analyze ESI component breakdowns and gravity characteristics
Review research priority scores and confidence intervals

5. System Visualization

Explore 3D representations of exoplanet systems
Analyze relationships between planetary characteristics and habitability
Compare multiple candidate systems

6. Data Export

Download full processed datasets
Export high-priority candidates for further research
Save Earth-like and high similarity candidates
Technical Architecture

## Data Processing Layer

CSV data ingestion and validation
Feature engineering and transformation
Earth Similarity Index calculations
Advanced metric computations
Machine Learning Layer

Ensemble model training and validation
Neural network integration
Cross-validation and performance evaluation
Real-time prediction engine
Presentation Layer

Streamlit web application framework
Interactive Plotly visualizations
Real-time filtering and parameter adjustment
Export functionality
Model Performance

## The ensemble AI system achieves:

ROC-AUC scores exceeding 0.95
Precision and recall balanced for exoplanet detection
Cross-validated performance with stratified testing
Confidence intervals for all predictions

Contributing

## Team members should follow these guidelines:

Use descriptive commit messages
Maintain code documentation
Test changes thoroughly before submission
Follow PEP 8 coding standards for Python
Update documentation for new features
License

## This project is proprietary and developed for research purposes. All rights reserved.

## Acknowledgments

NASA Exoplanet Archive for data sources
Kepler Mission science team
Scikit-learn, XGBoost, and LightGBM development teams
Streamlit and Plotly visualization libraries
Support

For technical support or questions regarding the Habitable Worlds Prototype, contact the core development team.
