#streamlit run ML.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="NASA Exoplanet Detection", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%); color: #ffffff;}
    .stButton>button {background: linear-gradient(45deg, #005288 0%, #0077cc 100%); color: white; border: none; border-radius: 8px; padding: 0.7em 1.5em; font-weight: 600;}
    .metric-container {background: rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 15px; margin: 8px; border: 1px solid rgba(255, 255, 255, 0.1);}
    .stSlider>div>div>div>div{background: linear-gradient(45deg, #005288 0%, #0077cc 100%);}
    .tab-container {background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; margin: 10px 0;}
    .header-accent {background: linear-gradient(45deg, #005288, #00a8ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;}
    .data-card {background: rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px; margin: 6px 0; border-left: 4px solid #005288;}
    .upload-section {background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; margin: 10px 0; border: 2px dashed #0077cc;}
    .esi-high {color: #51cf66; font-weight: bold;}
    .esi-medium {color: #fcc419; font-weight: bold;}
    .esi-low {color: #ff6b6b; font-weight: bold;}
    .gravity-high {color: #ff6b6b; font-weight: bold;}
    .gravity-earth {color: #51cf66; font-weight: bold;}
    .gravity-low {color: #fcc419; font-weight: bold;}
    .top-candidate {background: linear-gradient(135deg, #005288, #0077cc); border-radius: 12px; padding: 20px; margin: 10px 0; border: 2px solid #00a8ff; box-shadow: 0 4px 15px rgba(0, 168, 255, 0.3);}
    .research-metric {background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 10px; padding: 15px; margin: 8px; border: 1px solid rgba(255, 255, 255, 0.1);}
    .ai-feature-card {background: linear-gradient(135deg, #005288, #0077cc); border-radius: 15px; padding: 25px; margin: 15px 0; border: 2px solid #00a8ff; box-shadow: 0 8px 25px rgba(0, 168, 255, 0.4);}
    .model-performance {background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; padding: 20px; margin: 10px; border: 1px solid rgba(0, 168, 255, 0.3);}
    .prediction-high {color: #51cf66; font-weight: bold; font-size: 1.1em;}
    .prediction-medium {color: #fcc419; font-weight: bold; font-size: 1.1em;}
    .prediction-low {color: #ff6b6b; font-weight: bold; font-size: 1.1em;}
    .shap-plot {background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

st.title("NASA Exoplanet Research Platform")
st.markdown("<div class='header-accent'>Advanced Machine Learning for Planetary Candidate Analysis & Earth Similarity Assessment</div>", unsafe_allow_html=True)

def calculate_enhanced_earth_similarity_index(df):
    try:
        earth_radius = 1.0
        earth_density = 5.51
        earth_temperature = 288.0
        earth_flux = 1.0
        earth_gravity = 1.0
        earth_albedo = 0.3
        
        df = df.copy()
        
        if 'koi_prad' in df.columns:
            radius_esi = 1 - np.abs(df['koi_prad'] - earth_radius) / (df['koi_prad'] + earth_radius)
            radius_esi = np.clip(radius_esi, 0, 1)
        else:
            radius_esi = np.zeros(len(df))
        
        if 'koi_teq' in df.columns:
            temp_esi = 1 - np.abs(df['koi_teq'] - earth_temperature) / (df['koi_teq'] + earth_temperature)
            temp_esi = np.clip(temp_esi, 0, 1)
        else:
            temp_esi = np.zeros(len(df))
        
        if 'koi_insol' in df.columns:
            flux_esi = 1 - np.abs(df['koi_insol'] - earth_flux) / (df['koi_insol'] + earth_flux)
            flux_esi = np.clip(flux_esi, 0, 1)
        else:
            flux_esi = np.zeros(len(df))
        
        if 'koi_prad' in df.columns:
            estimated_mass = df['koi_prad'] ** 2.06
            estimated_gravity = estimated_mass / (df['koi_prad'] ** 2)
            gravity_ratio = estimated_gravity / earth_gravity
            gravity_esi = 1 - np.abs(gravity_ratio - 1) / (gravity_ratio + 1)
            gravity_esi = np.clip(gravity_esi, 0, 1)
            
            df['estimated_mass'] = estimated_mass
            df['estimated_gravity'] = estimated_gravity
            df['gravity_ratio'] = gravity_ratio
            
            df['escape_velocity'] = np.sqrt(2 * 6.67430e-11 * estimated_mass * 5.972e24 / (df['koi_prad'] * 6371000))
        else:
            gravity_esi = np.zeros(len(df))
            df['estimated_mass'] = 0.0
            df['estimated_gravity'] = 0.0
            df['gravity_ratio'] = 0.0
            df['escape_velocity'] = 0.0
        
        if 'koi_period' in df.columns and 'koi_prad' in df.columns:
            df['orbital_velocity'] = (2 * np.pi * 1.496e11) / (df['koi_period'] * 86400)
            df['hill_sphere'] = df['koi_prad'] * (df['estimated_mass'] / (3 * 1))**(1/3)
        else:
            df['orbital_velocity'] = 0.0
            df['hill_sphere'] = 0.0
        
        esi_components = []
        if 'koi_prad' in df.columns:
            esi_components.append(radius_esi)
        if 'koi_teq' in df.columns:
            esi_components.append(temp_esi)
        if 'koi_insol' in df.columns:
            esi_components.append(flux_esi)
        if 'koi_prad' in df.columns:
            esi_components.append(gravity_esi)
        
        if len(esi_components) > 0:
            esi_array = np.column_stack(esi_components)
            df['earth_similarity_index'] = np.prod(esi_array, axis=1) ** (1/len(esi_components))
            
            df['radius_esi'] = radius_esi
            df['temp_esi'] = temp_esi
            df['flux_esi'] = flux_esi
            df['gravity_esi'] = gravity_esi
        else:
            df['earth_similarity_index'] = 0.0
        
        conditions = [
            df['earth_similarity_index'] >= 0.8,
            df['earth_similarity_index'] >= 0.6,
            df['earth_similarity_index'] >= 0.4,
            df['earth_similarity_index'] < 0.4
        ]
        choices = ['Earth-like', 'Highly Similar', 'Moderately Similar', 'Low Similarity']
        df['esi_category'] = np.select(conditions, choices, default='Unknown')
        
        gravity_conditions = [
            df['gravity_ratio'] >= 2.0,
            (df['gravity_ratio'] >= 0.8) & (df['gravity_ratio'] < 2.0),
            (df['gravity_ratio'] >= 0.5) & (df['gravity_ratio'] < 0.8),
            df['gravity_ratio'] < 0.5
        ]
        gravity_choices = ['High Gravity', 'Earth-like Gravity', 'Low Gravity', 'Very Low Gravity']
        df['gravity_category'] = np.select(gravity_conditions, gravity_choices, default='Unknown')
        
        if 'habitable_zone_flag' in df.columns:
            esi_component = np.clip(df['earth_similarity_index'], 0, 1)
            hab_zone_component = np.clip(df['habitable_zone_flag'], 0, 1)
            gravity_esi_component = np.clip(df['gravity_esi'], 0, 1)
            gravity_similarity = np.clip(1 - np.abs(df['gravity_ratio'] - 1), 0, 1)
            radius_similarity = np.clip(1 - np.abs(df.get('koi_prad', 1) - 1), 0, 1)
            
            df['habitability_score'] = (
                esi_component * 0.4 + 
                hab_zone_component * 0.2 +
                gravity_esi_component * 0.2 +
                gravity_similarity * 0.1 +
                radius_similarity * 0.1
            )
        else:
            df['habitability_score'] = df['earth_similarity_index'] * 0.6 + df['gravity_esi'] * 0.4
        
        habitability_component = np.clip(df['habitability_score'], 0, 1)
        
        if 'koi_model_snr' in df.columns:
            snr_data = df['koi_model_snr'].fillna(0)
            if snr_data.max() > snr_data.min():
                snr_normalized = (snr_data - snr_data.min()) / (snr_data.max() - snr_data.min())
            else:
                snr_normalized = 0.5  # Default if all values are the same
        else:
            snr_normalized = 0.5  # Default if column doesn't exist
        
        df['research_priority'] = (
            habitability_component * 0.7 + 
            np.clip(snr_normalized, 0, 1) * 0.3
        )
        
        df['research_priority'] = np.clip(df['research_priority'], 0, 1)
        df['habitability_score'] = np.clip(df['habitability_score'], 0, 1)
        
        return df
        
    except Exception as e:
        st.warning(f"Could not calculate Enhanced Earth Similarity Index: {str(e)}")
        df['earth_similarity_index'] = 0.0
        df['esi_category'] = 'Unknown'
        df['habitability_score'] = 0.0
        df['gravity_ratio'] = 0.0
        df['gravity_category'] = 'Unknown'
        df['research_priority'] = 0.0
        return df

def calculate_advanced_metrics(df):
    try:
        if 'koi_steff' in df.columns and 'koi_srad' in df.columns:
            df['stellar_luminosity'] = 4 * np.pi * (df['koi_srad'] * 6.957e8)**2 * 5.67e-8 * df['koi_steff']**4
            
        if 'koi_period' in df.columns and 'stellar_luminosity' in df.columns:
            df['equilibrium_temperature'] = 278 * (df['stellar_luminosity']**0.25) / (df['koi_period']**0.5)
            
        if 'koi_prad' in df.columns and 'estimated_mass' in df.columns:
            df['bulk_density'] = df['estimated_mass'] * 5.51 / (df['koi_prad']**3)
            
        if 'koi_insol' in df.columns:
            df['greenhouse_effect'] = df['koi_teq'] / (278 * df['koi_insol']**0.25)
            
        return df
    except Exception as e:
        st.warning(f"Could not calculate advanced metrics: {str(e)}")
        return df

def process_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, comment='#')
        
        required_cols = ['kepid', 'koi_disposition', 'koi_period', 'koi_prad', 'koi_teq']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None, []
        
        available_cols = ['kepid','kepoi_name','kepler_name','koi_disposition','koi_period','koi_prad','koi_teq',
                         'koi_insol','koi_model_snr','koi_steff','koi_slogg','koi_srad','ra','dec','koi_kepmag']
        available_cols = [col for col in available_cols if col in df.columns]
        df = df[available_cols].copy()
        
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'koi_disposition':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        numeric_cols = ['koi_prad', 'koi_teq', 'koi_period', 'koi_insol', 'koi_steff', 'koi_srad']
        available_numeric = [col for col in numeric_cols if col in df.columns]
        df.dropna(subset=available_numeric, how='all', inplace=True)
        
        if len(df) == 0:
            st.error("No data remaining after cleaning. Please check your file.")
            return None, []
        
        df['label'] = df['koi_disposition'].map({'CONFIRMED':1, 'FALSE POSITIVE':0, 'CANDIDATE':-1})
        df['label'].fillna(-1, inplace=True)
        
        if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            df['planet_radius_km'] = df['koi_prad'] * 6371
            df['star_radius_km'] = df['koi_srad'] * 695700
            df['radius_to_star_ratio'] = df['planet_radius_km'] / df['star_radius_km']
            df['planet_area_ratio'] = (df['planet_radius_km']**2) / (df['star_radius_km']**2)
        
        if 'koi_insol' in df.columns and 'koi_teq' in df.columns:
            df['flux_ratio'] = df['koi_insol'] / df['koi_teq']
        
        if 'koi_teq' in df.columns and 'koi_steff' in df.columns:
            df['eqtemp_to_star_ratio'] = df['koi_teq'] / df['koi_steff']
        
        if 'koi_period' in df.columns:
            df['semi_major_axis_au'] = (df['koi_period']**2)**(1/3)
        
        if 'koi_prad' in df.columns and 'koi_teq' in df.columns:
            df['planet_density_proxy'] = df['koi_prad'] / df['koi_teq']
        
        if 'koi_teq' in df.columns:
            df['habitable_zone_flag'] = ((df['koi_teq'] >= 200) & (df['koi_teq'] <= 350)).astype(int)
        
        if 'planet_area_ratio' in df.columns:
            df['transit_depth'] = df['planet_area_ratio'] * 1e6
        
        if 'koi_period' in df.columns and 'radius_to_star_ratio' in df.columns:
            df['transit_duration'] = (df['koi_period'] * df['radius_to_star_ratio']) / (2 * np.pi)
        
        df = calculate_enhanced_earth_similarity_index(df)
        df = calculate_advanced_metrics(df)
        
        scalable_features = []
        feature_candidates = ['koi_period','koi_prad','koi_teq','koi_insol','koi_steff','koi_srad',
                             'radius_to_star_ratio','planet_area_ratio','flux_ratio','eqtemp_to_star_ratio',
                             'semi_major_axis_au','planet_density_proxy','transit_depth','transit_duration',
                             'earth_similarity_index','habitability_score','estimated_gravity','gravity_ratio',
                             'research_priority','stellar_luminosity','bulk_density','greenhouse_effect']
        
        for feature in feature_candidates:
            if feature in df.columns:
                scalable_features.append(feature)
        
        if scalable_features:
            for feature in scalable_features:
                if feature in df.columns:
                    mask = df[feature].notna()
                    if mask.sum() > 0:
                        valid_data = df.loc[mask, feature]
                        scaler = RobustScaler()
                        try:
                            scaled_values = scaler.fit_transform(valid_data.values.reshape(-1, 1))
                            df.loc[mask, f'{feature}_scaled'] = scaled_values.flatten()
                        except Exception as e:
                            continue
        
        for col in ['koi_period', 'koi_prad', 'koi_teq']:
            if col in df.columns:
                df[f'{col}_original'] = df[col].copy()
        
        return df, scalable_features
        
    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        return None, []

st.sidebar.header("Data Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Kepler Data CSV", type=['csv'])

if uploaded_file is None:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.subheader("Upload Kepler Mission Data")
    st.write("Please upload a CSV file containing Kepler exoplanet data to begin analysis.")
    st.write("Required columns: kepid, koi_disposition, koi_period, koi_prad, koi_teq")
    st.info("You can download sample data from NASA Exoplanet Archive")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

try:
    with st.spinner("Processing exoplanet data and calculating Earth Similarity..."):
        df, scalable_features = process_data(uploaded_file)
        
    if df is None or len(df) == 0:
        st.error("No valid data found. Please check your file format.")
        st.stop()
    
    st.success(f"Successfully loaded {len(df)} exoplanet records with Enhanced Earth Similarity Analysis")
    
    st.sidebar.header("Discovery Parameters")
    st.sidebar.markdown("Adjust filters to refine candidate search")

    slider_config = {}
    
    if 'koi_period_original' in df.columns:
        period_data = df['koi_period_original'].dropna()
        if len(period_data) > 0:
            period_min, period_max = float(period_data.min()), float(period_data.max())
            period_default = (float(period_data.quantile(0.1)), float(period_data.quantile(0.9)))
            slider_config['period'] = {
                'min': period_min, 'max': period_max, 'default': period_default,
                'label': f"Orbital Period Range ({period_min:.1f} to {period_max:.1f} days)"
            }

    if 'koi_teq_original' in df.columns:
        teq_data = df['koi_teq_original'].dropna()
        if len(teq_data) > 0:
            teq_min, teq_max = float(teq_data.min()), float(teq_data.max())
            teq_default = (float(teq_data.quantile(0.1)), float(teq_data.quantile(0.9)))
            slider_config['teq'] = {
                'min': teq_min, 'max': teq_max, 'default': teq_default,
                'label': f"Temperature Range ({teq_min:.0f} to {teq_max:.0f} K)"
            }

    if 'koi_prad_original' in df.columns:
        radius_data = df['koi_prad_original'].dropna()
        if len(radius_data) > 0:
            radius_min, radius_max = float(radius_data.min()), float(radius_data.max())
            radius_default = (float(radius_data.quantile(0.1)), float(radius_data.quantile(0.9)))
            slider_config['radius'] = {
                'min': radius_min, 'max': radius_max, 'default': radius_default,
                'label': f"Planet Radius Range ({radius_min:.1f} to {radius_max:.1f} Earth Radii)"
            }

    if 'period' in slider_config:
        period_range = st.sidebar.slider(
            slider_config['period']['label'],
            slider_config['period']['min'],
            slider_config['period']['max'],
            slider_config['period']['default']
        )

    if 'teq' in slider_config:
        teq_range = st.sidebar.slider(
            slider_config['teq']['label'],
            slider_config['teq']['min'],
            slider_config['teq']['max'],
            slider_config['teq']['default']
        )

    if 'radius' in slider_config:
        radius_range = st.sidebar.slider(
            slider_config['radius']['label'],
            slider_config['radius']['min'],
            slider_config['radius']['max'],
            slider_config['radius']['default']
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Earth Similarity Filters")
    
    esi_threshold = st.sidebar.slider(
        "Minimum Earth Similarity Index (ESI)",
        0.0, 1.0, 0.5, 0.05,
        help="Filter candidates by their similarity to Earth (0 = completely different, 1 = identical to Earth)"
    )
    
    gravity_threshold = st.sidebar.slider(
        "Gravity Range (Earth = 1.0)",
        0.1, 3.0, (0.5, 1.5), 0.1,
        help="Filter by estimated surface gravity relative to Earth"
    )

    habitable_only = st.sidebar.checkbox("Show Only Habitable Zone Candidates")
    high_confidence = st.sidebar.checkbox("High Confidence Candidates Only")
    earth_like_only = st.sidebar.checkbox("Show Only Earth-like Candidates (ESI > 0.8)")
    research_priority_only = st.sidebar.checkbox("Show Only High Research Priority")

    filtered_df = df.copy()
    
    if 'period' in slider_config:
        filtered_df = filtered_df[
            (filtered_df['koi_period_original'] >= period_range[0]) & 
            (filtered_df['koi_period_original'] <= period_range[1])
        ]
    
    if 'teq' in slider_config:
        filtered_df = filtered_df[
            (filtered_df['koi_teq_original'] >= teq_range[0]) & 
            (filtered_df['koi_teq_original'] <= teq_range[1])
        ]
    
    if 'radius' in slider_config:
        filtered_df = filtered_df[
            (filtered_df['koi_prad_original'] >= radius_range[0]) & 
            (filtered_df['koi_prad_original'] <= radius_range[1])
        ]

    if 'earth_similarity_index' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['earth_similarity_index'] >= esi_threshold]

    if 'gravity_ratio' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['gravity_ratio'] >= gravity_threshold[0]) & 
            (filtered_df['gravity_ratio'] <= gravity_threshold[1])
        ]

    if earth_like_only and 'earth_similarity_index' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['earth_similarity_index'] > 0.8]

    if habitable_only and 'habitable_zone_flag' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['habitable_zone_flag'] == 1]

    if research_priority_only and 'research_priority' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['research_priority'] > 0.7]

    st.sidebar.markdown(f"Filtered Results: {len(filtered_df)} candidates")
    
    if 'earth_similarity_index' in filtered_df.columns:
        avg_esi = filtered_df['earth_similarity_index'].mean()
        max_esi = filtered_df['earth_similarity_index'].max()
        earth_like_count = len(filtered_df[filtered_df['earth_similarity_index'] > 0.8])
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("ESI Statistics")
        st.sidebar.metric("Average ESI", f"{avg_esi:.3f}")
        st.sidebar.metric("Maximum ESI", f"{max_esi:.3f}")
        st.sidebar.metric("Earth-like Candidates", earth_like_count)

    labeled_df = filtered_df[filtered_df['label'].notna() & (filtered_df['label'] != -1)]
    candidate_df = filtered_df[(filtered_df['label'].isna()) | (filtered_df['label'] == -1)]
    
    scaled_features = [f + '_scaled' for f in scalable_features]
    features = [f for f in scaled_features if f in filtered_df.columns]
    if 'habitable_zone_flag' in filtered_df.columns:
        features.append('habitable_zone_flag')

    # Initialize model variables
    model_trained = False
    ensemble_clf = None
    X_test, y_test, y_pred, y_proba = None, None, None, None
    cv_scores = None
    feature_importance = None

    # Train the model if we have enough data
    if len(labeled_df) > 10 and len(features) > 3:
        X = labeled_df[features].dropna()
        y = labeled_df.loc[X.index, 'label']

        if len(X) > 10:
            with st.spinner("Training advanced ensemble model for exoplanet detection..."):
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
                gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
                ensemble_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft', weights=[2, 1])
                ensemble_clf.fit(X, y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                y_pred = ensemble_clf.predict(X_test)
                y_proba = ensemble_clf.predict_proba(X_test)[:, 1]

                cv_scores = cross_val_score(ensemble_clf, X, y, cv=min(3, len(X)//3), scoring='roc_auc')
                
                rf.fit(X, y)
                feature_importance = rf.feature_importances_
                
                model_trained = True

            if not candidate_df.empty and len(features) > 0:
                X_candidates_raw = candidate_df[features]
                valid_mask = X_candidates_raw.notnull().all(axis=1)
                candidate_indices = candidate_df.index[valid_mask]
                X_candidates = X_candidates_raw.loc[valid_mask].copy()
                
                if len(X_candidates) > 0:
                    rf.fit(X, y)
                    gb.fit(X, y)
                    
                    rf_proba = rf.predict_proba(X_candidates)[:, 1]
                    gb_proba = gb.predict_proba(X_candidates)[:, 1]
                    
                    candidate_probs = np.column_stack([rf_proba, gb_proba])
                    mean_prob = candidate_probs.mean(axis=1)
                    std_prob = candidate_probs.std(axis=1)
                    
                    candidate_df = candidate_df.copy()
                    candidate_df.loc[candidate_indices, 'predicted_prob'] = mean_prob
                    candidate_df.loc[candidate_indices, 'uncertainty'] = std_prob
                    candidate_df.loc[candidate_indices, 'confidence_lower'] = np.maximum(0, mean_prob - 1.96 * std_prob)
                    candidate_df.loc[candidate_indices, 'confidence_upper'] = np.minimum(1, mean_prob + 1.96 * std_prob)
                    
                    candidate_df = candidate_df.sort_values(by='earth_similarity_index', ascending=False)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Research Dashboard", "AI Exoplanet Detection", "Candidate Explorer", "System Visualizer", "Advanced Analytics", "Data Export"])

    with tab1:
        st.header("Research Dashboard")
        
        if model_trained:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, y_proba):.4f}")
            with col2:
                st.metric("Cross-Validation Score", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            with col3:
                st.metric("Confirmed Exoplanets", f"{len(labeled_df[labeled_df['label'] == 1])}")
            with col4:
                st.metric("Training Samples", f"{len(X)}")
            
            st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Research Metrics")
                
                research_cols = st.columns(3)
                with research_cols[0]:
                    if 'earth_similarity_index' in df.columns:
                        st.markdown("<div class='research-metric'>", unsafe_allow_html=True)
                        st.metric("Average ESI", f"{df['earth_similarity_index'].mean():.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with research_cols[1]:
                    if 'habitability_score' in df.columns:
                        st.markdown("<div class='research-metric'>", unsafe_allow_html=True)
                        st.metric("Avg Habitability", f"{df['habitability_score'].mean():.3f}")
                        st.metric("High Habitability", f"{len(df[df['habitability_score'] > 0.7])}")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with research_cols[2]:
                    if 'research_priority' in df.columns:
                        st.markdown("<div class='research-metric'>", unsafe_allow_html=True)
                        st.metric("Research Priority", f"{df['research_priority'].mean():.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                st.subheader("Classification Performance")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap='Blues'))
                
            with col2:
                st.subheader("Feature Importance")
                
                try:
                    feature_names_clean = []
                    for f in X.columns:
                        if f == 'habitable_zone_flag':
                            feature_names_clean.append(f)
                        else:
                            clean_name = f.replace('_scaled', '').replace('_original', '')
                            feature_names_clean.append(clean_name)
                    
                    importance_df = pd.DataFrame({
                        'feature': feature_names_clean,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=True)
                    
                    fig_importance = go.Figure()
                    
                    fig_importance.add_trace(go.Bar(
                        y=importance_df['feature'],
                        x=importance_df['importance'],
                        orientation='h',
                        marker_color='#005288'
                    ))
                    
                    fig_importance.update_layout(
                        title="Feature Importance (Random Forest)",
                        xaxis_title="Importance Score",
                        yaxis_title="Features",
                        height=400
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Feature importance calculation failed: {str(e)}")
            
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['False Positive', 'Confirmed'],
                              y=['False Positive', 'Confirmed'])
            fig_cm.update_layout(title="Model Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
                    
        else:
            st.warning("Insufficient labeled data for model training")
            st.info("Upload a dataset with confirmed exoplanets and false positives to enable AI detection capabilities")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.header("AI Exoplanet Detection Engine")
        st.markdown("<div class='header-accent'>Advanced Machine Learning for Automated Planetary Candidate Validation</div>", unsafe_allow_html=True)
        
        if model_trained:
            st.markdown("<div class='ai-feature-card'>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Accuracy", f"{roc_auc_score(y_test, y_proba):.3f}")
            with col2:
                st.metric("Cross-Validation", f"{cv_scores.mean():.3f}")
            with col3:
                st.metric("Precision", f"{classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.3f}")
            with col4:
                st.metric("Recall", f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Architecture")
                st.markdown("""
                **Ensemble Learning Approach:**
                - **Random Forest Classifier**: 100 trees with balanced class weights
                - **Gradient Boosting**: 100 estimators with adaptive learning
                - **Voting Ensemble**: Soft voting with weighted probabilities
                - **Feature Scaling**: Robust scaling for outlier resistance
                
                **Training Data:**
                - Uses confirmed exoplanets vs false positives
                - Cross-validation with stratification
                - Balanced class weighting
                - Feature importance analysis
                """)
                
                st.subheader("Performance Metrics")
                if model_trained:
                    report = classification_report(y_test, y_pred, output_dict=True)
                    metrics_df = pd.DataFrame({
                        'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        'Score': [
                            report['1']['precision'],
                            report['1']['recall'], 
                            report['1']['f1-score'],
                            roc_auc_score(y_test, y_proba)
                        ]
                    })
                    
                    fig_metrics = px.bar(metrics_df, x='Metric', y='Score', 
                                       title="Model Performance Metrics",
                                       color='Score',
                                       color_continuous_scale='Viridis')
                    fig_metrics.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig_metrics, use_container_width=True)
            
            with col2:
                st.subheader("Real-time Predictions")
                
                if not candidate_df.empty and 'predicted_prob' in candidate_df.columns:
                    top_predictions = candidate_df.nlargest(8, 'predicted_prob')
                    
                    st.markdown("**Top AI-Validated Candidates:**")
                    
                    for idx, (_, row) in enumerate(top_predictions.head(4).iterrows()):
                        prob = row['predicted_prob']
                        if prob >= 0.8:
                            prob_class = "prediction-high"
                            confidence = "High Confidence"
                        elif prob >= 0.6:
                            prob_class = "prediction-medium"
                            confidence = "Medium Confidence"
                        else:
                            prob_class = "prediction-low"
                            confidence = "Low Confidence"
                        
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            st.write(f"**KEP-{int(row['kepid'])}**")
                            if 'koi_prad_original' in row:
                                st.write(f"Radius: {row['koi_prad_original']:.2f} Earth")
                            if 'koi_teq_original' in row:
                                st.write(f"Temp: {row['koi_teq_original']:.0f} K")
                        with col_b:
                            st.markdown(f"<div class='{prob_class}'>Probability: {prob:.3f}</div>", unsafe_allow_html=True)
                            st.write(confidence)
                        
                        st.progress(float(prob))
                        st.markdown("---")
                
                st.subheader("Prediction Confidence")
                if not candidate_df.empty and 'predicted_prob' in candidate_df.columns:
                    fig_confidence = px.histogram(
                        candidate_df,
                        x='predicted_prob',
                        nbins=20,
                        title="Prediction Probability Distribution",
                        color_discrete_sequence=['#00a8ff']
                    )
                    fig_confidence.add_vline(x=0.8, line_dash="dash", line_color="green", 
                                           annotation_text="High Confidence")
                    fig_confidence.add_vline(x=0.6, line_dash="dash", line_color="orange", 
                                           annotation_text="Medium Confidence")
                    st.plotly_chart(fig_confidence, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
            st.subheader("Explainable AI - Feature Importance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if model_trained:
                    feature_names_clean = []
                    for f in X.columns:
                        if f == 'habitable_zone_flag':
                            feature_names_clean.append(f)
                        else:
                            clean_name = f.replace('_scaled', '').replace('_original', '')
                            feature_names_clean.append(clean_name)
                    
                    importance_df = pd.DataFrame({
                        'feature': feature_names_clean,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=True)
                    
                    fig_importance_detailed = go.Figure()
                    
                    fig_importance_detailed.add_trace(go.Bar(
                        y=importance_df['feature'],
                        x=importance_df['importance'],
                        orientation='h',
                        marker_color='#005288',
                        text=importance_df['importance'].round(3),
                        textposition='auto'
                    ))
                    
                    fig_importance_detailed.update_layout(
                        title="Detailed Feature Importance Analysis",
                        xaxis_title="Importance Score",
                        yaxis_title="Features",
                        height=500
                    )
                    st.plotly_chart(fig_importance_detailed, use_container_width=True)
            
            with col2:
                st.subheader("Model Learning Curve")
                st.markdown("""
                **Training Characteristics:**
                - **Ensemble Method**: Combines multiple algorithms for robust predictions
                - **Feature Selection**: Automatically identifies most predictive features  
                - **Cross-Validation**: 3-fold validation ensures generalization
                - **Class Balancing**: Handles imbalanced exoplanet datasets
                
                **Key Advantages:**
                - Higher accuracy than single models
                - Reduced overfitting through ensemble voting
                - Better handling of noisy Kepler data
                - Interpretable feature importance
                """)
                
                if model_trained:
                    st.subheader("Prediction Statistics")
                    high_conf_count = len(candidate_df[candidate_df['predicted_prob'] > 0.8]) if 'predicted_prob' in candidate_df.columns else 0
                    medium_conf_count = len(candidate_df[(candidate_df['predicted_prob'] > 0.6) & (candidate_df['predicted_prob'] <= 0.8)]) if 'predicted_prob' in candidate_df.columns else 0
                    
                    stats_df = pd.DataFrame({
                        'Confidence Level': ['High (>0.8)', 'Medium (0.6-0.8)', 'Low (<0.6)'],
                        'Count': [high_conf_count, medium_conf_count, len(candidate_df) - high_conf_count - medium_conf_count]
                    })
                    
                    fig_stats = px.pie(stats_df, values='Count', names='Confidence Level',
                                     title="Candidate Prediction Confidence Distribution",
                                     color_discrete_sequence=['#51cf66', '#fcc419', '#ff6b6b'])
                    st.plotly_chart(fig_stats, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.markdown("<div class='ai-feature-card'>", unsafe_allow_html=True)
            st.warning("AI Detection Engine Requires Training Data")
            st.info("""
            To enable the AI Exoplanet Detection capabilities:
            
            1. **Upload a dataset** with both confirmed exoplanets and false positives
            2. **Ensure sufficient labeled data** (minimum 10 samples each)
            3. **Include key features** like orbital period, planetary radius, and temperature
            
            The AI engine will automatically train an ensemble model to distinguish between
            real exoplanets and false positives with high accuracy.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.header("Enhanced Earth Similarity Analysis")
        
        if not candidate_df.empty and 'earth_similarity_index' in candidate_df.columns:
            top_candidates_by_esi = candidate_df.nlargest(10, 'earth_similarity_index')
            
            if high_confidence and 'uncertainty' in top_candidates_by_esi.columns:
                top_candidates_by_esi = top_candidates_by_esi[top_candidates_by_esi['uncertainty'] < 0.1]
            
            st.subheader("Top 5 Highest Earth Similarity Index Planets")
            
            top_5_esi = candidate_df.nlargest(5, 'earth_similarity_index')
            
            cols = st.columns(5)
            for idx, (col, (_, row)) in enumerate(zip(cols, top_5_esi.iterrows())):
                with col:
                    esi_value = row['earth_similarity_index']
                    gravity_ratio = row.get('gravity_ratio', 0)
                    research_priority = row.get('research_priority', 0)
                    
                    if esi_value >= 0.8:
                        esi_class = "esi-high"
                        esi_label = "Earth-like"
                    elif esi_value >= 0.6:
                        esi_class = "esi-medium"
                        esi_label = "Highly Similar"
                    else:
                        esi_class = "esi-low"
                        esi_label = "Moderately Similar"
                    
                    if gravity_ratio >= 1.5:
                        gravity_class = "gravity-high"
                        gravity_label = "High Gravity"
                    elif gravity_ratio >= 0.8 and gravity_ratio <= 1.2:
                        gravity_class = "gravity-earth"
                        gravity_label = "Earth-like"
                    else:
                        gravity_class = "gravity-low"
                        gravity_label = "Low Gravity"
                    
                    candidate_html = f"""
                    <div class='top-candidate'>
                        <h3>#{idx+1}</h3>
                        <h4>KEP-{int(row['kepid'])}</h4>
                        <p><b>ESI:</b> <span class='{esi_class}'>{esi_value:.3f}</span></p>
                        <p><b>Category:</b> {esi_label}</p>
                        <p><b>Gravity:</b> <span class='{gravity_class}'>{gravity_ratio:.2f}g</span></p>
                        <p><b>Research Priority:</b> {research_priority:.3f}</p>
                    """
                    if 'koi_period_original' in row:
                        candidate_html += f"<p><b>Period:</b> {row['koi_period_original']:.2f} days</p>"
                    if 'koi_prad_original' in row:
                        candidate_html += f"<p><b>Radius:</b> {row['koi_prad_original']:.2f} Earth</p>"
                    if 'koi_teq_original' in row:
                        candidate_html += f"<p><b>Temp:</b> {row['koi_teq_original']:.0f} K</p>"
                    if 'habitable_zone_flag' in row:
                        candidate_html += f"<p><b>Habitable:</b> {'Yes' if row['habitable_zone_flag'] else 'No'}</p>"
                    
                    candidate_html += "</div>"
                    st.markdown(candidate_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candidates", len(candidate_df))
            with col2:
                st.metric("High Probability", len(top_candidates_by_esi[top_candidates_by_esi['predicted_prob'] > 0.8]) if 'predicted_prob' in top_candidates_by_esi.columns else 0)
            with col3:
                habitable_count = top_candidates_by_esi['habitable_zone_flag'].sum() if 'habitable_zone_flag' in top_candidates_by_esi.columns else 0
                st.metric("In Habitable Zone", habitable_count)
            with col4:
                if 'earth_similarity_index' in top_candidates_by_esi.columns:
                    avg_esi = top_candidates_by_esi['earth_similarity_index'].mean()
                    st.metric("Average ESI", f"{avg_esi:.3f}")
            
            st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Enhanced Earth Similarity Analysis")
                
                if 'earth_similarity_index' in candidate_df.columns:
                    st.subheader("ESI Component Analysis")
                    
                    component_data = []
                    for idx, row in top_candidates_by_esi.head(10).iterrows():
                        component_data.append({
                            'KEPID': f"KEP-{int(row['kepid'])}",
                            'Radius ESI': row.get('radius_esi', 0),
                            'Temperature ESI': row.get('temp_esi', 0),
                            'Flux ESI': row.get('flux_esi', 0),
                            'Gravity ESI': row.get('gravity_esi', 0),
                            'Total ESI': row['earth_similarity_index']
                        })
                    
                    component_df = pd.DataFrame(component_data)
                    fig_components = px.bar(component_df, 
                                          x='KEPID', 
                                          y=['Radius ESI', 'Temperature ESI', 'Flux ESI', 'Gravity ESI'],
                                          title="ESI Component Breakdown for Top 10 Candidates by ESI",
                                          color_discrete_map={
                                              'Radius ESI': '#FF6B6B',
                                              'Temperature ESI': '#4ECDC4', 
                                              'Flux ESI': '#45B7D1',
                                              'Gravity ESI': '#96CEB4'
                                          })
                    fig_components.update_layout(barmode='stack', height=400)
                    st.plotly_chart(fig_components, use_container_width=True)
                    
                    st.subheader("Gravity vs Earth Similarity")
                    fig_gravity_esi = px.scatter(
                        top_candidates_by_esi,
                        x='gravity_ratio',
                        y='earth_similarity_index',
                        size='koi_prad_original' if 'koi_prad_original' in top_candidates_by_esi.columns else None,
                        color='research_priority' if 'research_priority' in top_candidates_by_esi.columns else 'habitable_zone_flag',
                        hover_data=['kepid'],
                        title="Surface Gravity vs Earth Similarity Index for Top 10 Candidates",
                        labels={
                            'gravity_ratio': 'Surface Gravity (Earth = 1.0)',
                            'earth_similarity_index': 'Earth Similarity Index',
                            'research_priority': 'Research Priority',
                            'habitable_zone_flag': 'In Habitable Zone'
                        }
                    )
                    fig_gravity_esi.add_vline(x=1.0, line_dash="dash", line_color="green", 
                                            annotation_text="Earth Gravity")
                    fig_gravity_esi.add_hline(y=0.8, line_dash="dash", line_color="blue", 
                                            annotation_text="Earth-like Threshold")
                    st.plotly_chart(fig_gravity_esi, use_container_width=True)
                    
                else:
                    st.info("Enhanced Earth Similarity Index not available for visualization")
            
            with col2:
                st.subheader("Research Priority Analysis")
                
                if 'research_priority' in candidate_df.columns:
                    st.metric("Average Research Priority", f"{candidate_df['research_priority'].mean():.3f}")
                    st.metric("High Priority Candidates", f"{len(candidate_df[candidate_df['research_priority'] > 0.7])}")
                    
                    fig_priority_dist = px.histogram(
                        candidate_df,
                        x='research_priority',
                        nbins=20,
                        title="Research Priority Distribution",
                        color_discrete_sequence=['#00a8ff']
                    )
                    st.plotly_chart(fig_priority_dist, use_container_width=True)
                
                st.subheader("Advanced Metrics")
                if 'bulk_density' in candidate_df.columns:
                    st.metric("Avg Bulk Density", f"{candidate_df['bulk_density'].mean():.2f} g/cm³")
                if 'greenhouse_effect' in candidate_df.columns:
                    st.metric("Avg Greenhouse Effect", f"{candidate_df['greenhouse_effect'].mean():.2f}")
                if 'escape_velocity' in candidate_df.columns:
                    st.metric("Avg Escape Velocity", f"{candidate_df['escape_velocity'].mean():.1f} km/s")
            
            st.subheader("Detailed Candidate Table - Ranked by ESI")
            display_cols = ['kepid', 'earth_similarity_index', 'esi_category', 'gravity_ratio', 'gravity_category']
            if 'predicted_prob' in candidate_df.columns:
                display_cols.extend(['predicted_prob', 'uncertainty'])
            if 'research_priority' in candidate_df.columns:
                display_cols.append('research_priority')
            optional_cols = ['koi_period_original', 'koi_prad_original', 'koi_teq_original', 'habitable_zone_flag']
            for col in optional_cols:
                if col in candidate_df.columns:
                    display_cols.append(col)
            
            display_df = top_candidates_by_esi[display_cols].head(20).copy()
            display_df.columns = [col.replace('_original', '') for col in display_df.columns]
            
            def color_esi(val):
                if val >= 0.8:
                    return 'background-color: #51cf66; color: white'
                elif val >= 0.6:
                    return 'background-color: #fcc419; color: black'
                elif val >= 0.4:
                    return 'background-color: #ff922b; color: white'
                else:
                    return 'background-color: #ff6b6b; color: white'
            
            def color_gravity(val):
                if val >= 1.5:
                    return 'background-color: #ff6b6b; color: white'
                elif val >= 0.8 and val <= 1.2:
                    return 'background-color: #51cf66; color: white'
                elif val >= 0.5:
                    return 'background-color: #fcc419; color: black'
                else:
                    return 'background-color: #ff922b; color: white'
            
            def color_priority(val):
                if val >= 0.8:
                    return 'background-color: #51cf66; color: white'
                elif val >= 0.6:
                    return 'background-color: #fcc419; color: black'
                elif val >= 0.4:
                    return 'background-color: #ff922b; color: white'
                else:
                    return 'background-color: #ff6b6b; color: white'
            
            if 'earth_similarity_index' in display_df.columns and 'gravity_ratio' in display_df.columns:
                styled_df = display_df.style.format({
                    'predicted_prob': '{:.3f}',
                    'uncertainty': '{:.3f}',
                    'earth_similarity_index': '{:.3f}',
                    'gravity_ratio': '{:.2f}',
                    'research_priority': '{:.3f}',
                    'koi_period': '{:.2f}',
                    'koi_prad': '{:.2f}',
                    'koi_teq': '{:.1f}'
                }).applymap(color_esi, subset=['earth_similarity_index']).applymap(color_gravity, subset=['gravity_ratio'])
                
                if 'research_priority' in display_df.columns:
                    styled_df = styled_df.applymap(color_priority, subset=['research_priority'])
            else:
                styled_df = display_df.style.format({
                    'predicted_prob': '{:.3f}',
                    'uncertainty': '{:.3f}',
                    'koi_period': '{:.2f}',
                    'koi_prad': '{:.2f}',
                    'koi_teq': '{:.1f}'
                })
            
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No candidate predictions available. Ensure sufficient training data and features.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.header("Enhanced System Visualization")
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        
        if not candidate_df.empty and 'earth_similarity_index' in candidate_df.columns:
            viz_df = candidate_df.nlargest(50, 'earth_similarity_index').copy()
            
            if 'koi_prad_original' in viz_df.columns:
                viz_df['size_scaled'] = np.log(viz_df['koi_prad_original'] + 1) * 15
            else:
                viz_df['size_scaled'] = 5
                
            if 'koi_period_original' in viz_df.columns:
                viz_df['orbit_scaled'] = np.log(viz_df['koi_period_original'] + 1) * 25
            else:
                viz_df['orbit_scaled'] = np.arange(len(viz_df))
            
            color_column = 'earth_similarity_index'
            color_title = 'Earth Similarity Index'
            
            st.subheader("3D Exoplanet System Map - Top Candidates by ESI")
            
            fig_3d = go.Figure()
            
            fig_3d.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(
                    size=30,
                    color='yellow',
                    opacity=0.8
                ),
                name='Central Star',
                hoverinfo='text',
                text=['Central Star']
            ))
            
            fig_3d.add_trace(go.Scatter3d(
                x=viz_df['orbit_scaled'],
                y=np.zeros(len(viz_df)),
                z=np.zeros(len(viz_df)),
                mode='markers',
                marker=dict(
                    size=viz_df['size_scaled'],
                    color=viz_df[color_column],
                    colorscale='Viridis',
                    colorbar=dict(title=color_title),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=[f"KEP-{kepid}<br>ESI: {esi:.3f}<br>Period: {period:.1f} days<br>Radius: {radius:.1f} Earth<br>Research Priority: {priority:.3f}" 
                      for kepid, esi, period, radius, priority in 
                      zip(viz_df['kepid'], viz_df['earth_similarity_index'], 
                          viz_df['koi_period_original'], viz_df['koi_prad_original'],
                          viz_df.get('research_priority', 0))],
                hoverinfo='text',
                name='Exoplanet Candidates'
            ))
            
            for orbit in np.linspace(viz_df['orbit_scaled'].min(), viz_df['orbit_scaled'].max(), 10):
                theta = np.linspace(0, 2*np.pi, 100)
                x = orbit * np.cos(theta)
                y = orbit * np.sin(theta)
                z = np.zeros(100)
                
                fig_3d.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.2)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig_3d.update_layout(
                title="3D Exoplanet System Visualization - Top 50 Candidates by ESI",
                scene=dict(
                    xaxis_title="Orbital Distance (scaled)",
                    yaxis_title="",
                    zaxis_title="",
                    bgcolor='rgba(5, 10, 20, 1)',
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'earth_similarity_index' in candidate_df.columns and 'koi_prad_original' in candidate_df.columns:
                    st.subheader("ESI vs Planetary Characteristics")
                    fig_esi_radius = px.scatter(
                        candidate_df.nlargest(50, 'earth_similarity_index'),
                        x='koi_prad_original',
                        y='earth_similarity_index',
                        size='research_priority' if 'research_priority' in candidate_df.columns else 'predicted_prob',
                        color='gravity_ratio' if 'gravity_ratio' in candidate_df.columns else None,
                        hover_data=['kepid'],
                        color_continuous_scale='Viridis',
                        title="Earth Similarity vs Planetary Size & Gravity (Top 50 by ESI)",
                        labels={
                            'koi_prad_original': 'Planet Radius (Earth Radii)',
                            'earth_similarity_index': 'Earth Similarity Index',
                            'gravity_ratio': 'Surface Gravity (Earth = 1.0)',
                            'research_priority': 'Research Priority'
                        }
                    )
                    fig_esi_radius.add_vline(x=1.0, line_dash="dash", line_color="green", 
                                           annotation_text="Earth Size")
                    fig_esi_radius.add_hline(y=0.8, line_dash="dash", line_color="blue", 
                                           annotation_text="Earth-like")
                    st.plotly_chart(fig_esi_radius, use_container_width=True)
            
            with col2:
                if 'earth_similarity_index' in candidate_df.columns and 'koi_teq_original' in candidate_df.columns:
                    st.subheader("Habitability Analysis")
                    fig_habitability = px.scatter(
                        candidate_df.nlargest(50, 'earth_similarity_index'),
                        x='koi_teq_original',
                        y='habitability_score' if 'habitability_score' in candidate_df.columns else 'earth_similarity_index',
                        size='koi_prad_original' if 'koi_prad_original' in candidate_df.columns else None,
                        color='research_priority' if 'research_priority' in candidate_df.columns else 'gravity_ratio',
                        hover_data=['kepid'],
                        color_continuous_scale='Viridis',
                        title="Temperature vs Habitability Score (Top 50 by ESI)",
                        labels={
                            'koi_teq_original': 'Temperature (K)',
                            'habitability_score': 'Habitability Score',
                            'research_priority': 'Research Priority',
                            'gravity_ratio': 'Surface Gravity'
                        }
                    )
                    fig_habitability.add_vrect(x0=200, x1=350, 
                                             annotation_text="Habitable Zone", 
                                             annotation_position="top left",
                                             fillcolor="green", opacity=0.2, line_width=0)
                    st.plotly_chart(fig_habitability, use_container_width=True)
            
            if 'earth_similarity_index' in candidate_df.columns:
                st.subheader("Top Earth-Like Candidates Comparison")
                top_5_esi = candidate_df.nlargest(5, 'earth_similarity_index')
                
                if len(top_5_esi) > 0:
                    categories = ['ESI', 'Gravity Match', 'Habitable Zone', 'Size Match', 'Temp Match', 'Research Priority']
                    
                    fig_radar = go.Figure()
                    
                    for idx, row in top_5_esi.iterrows():
                        values = [
                            row['earth_similarity_index'],
                            row.get('gravity_esi', 0.5),
                            row['habitable_zone_flag'] if 'habitable_zone_flag' in row else 0,
                            1 - min(1, abs(row['koi_prad_original'] - 1) / 10) if 'koi_prad_original' in row else 0,
                            1 - min(1, abs(row['koi_teq_original'] - 288) / 500) if 'koi_teq_original' in row else 0,
                            row.get('research_priority', 0.5)
                        ]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=f"KEP-{int(row['kepid'])} (ESI: {row['earth_similarity_index']:.2f})"
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1])
                        ),
                        showlegend=True,
                        title="Enhanced Earth-Like Candidate Comparison - Top 5 by ESI",
                        height=500
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Visualizations require candidate predictions with proper feature data.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        st.header("Advanced Analytics")
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        
        if not candidate_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Statistical Analysis")
                
                if 'earth_similarity_index' in candidate_df.columns:
                    esi_stats = candidate_df['earth_similarity_index'].describe()
                    st.write("ESI Distribution Statistics:")
                    st.dataframe(esi_stats)
                    
                    st.subheader("Correlation Analysis")
                    
                    corr_cols = ['earth_similarity_index', 'gravity_ratio', 'koi_prad_original', 'koi_teq_original', 'koi_period_original']
                    available_corr = [col for col in corr_cols if col in candidate_df.columns]
                    
                    if len(available_corr) > 1:
                        corr_matrix = candidate_df[available_corr].corr()
                        fig_corr = px.imshow(corr_matrix, 
                                           text_auto=True, 
                                           color_continuous_scale='RdBu_r',
                                           aspect="auto")
                        fig_corr.update_layout(title="Feature Correlation Matrix")
                        st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                st.subheader("Advanced Metrics Distribution")
                
                if 'bulk_density' in candidate_df.columns:
                    fig_density = px.histogram(
                        candidate_df,
                        x='bulk_density',
                        nbins=20,
                        title="Bulk Density Distribution",
                        color_discrete_sequence=['#00a8ff']
                    )
                    fig_density.add_vline(x=5.51, line_dash="dash", line_color="green", 
                                        annotation_text="Earth Density")
                    st.plotly_chart(fig_density, use_container_width=True)
                
                if 'greenhouse_effect' in candidate_df.columns:
                    fig_greenhouse = px.histogram(
                        candidate_df,
                        x='greenhouse_effect',
                        nbins=20,
                        title="Greenhouse Effect Distribution",
                        color_discrete_sequence=['#ff6b6b']
                    )
                    fig_greenhouse.add_vline(x=1.0, line_dash="dash", line_color="green", 
                                           annotation_text="Earth-like")
                    st.plotly_chart(fig_greenhouse, use_container_width=True)
            
            st.subheader("Research Priority Analysis")
            
            if 'research_priority' in candidate_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    high_priority = candidate_df[candidate_df['research_priority'] > 0.7]
                    st.metric("High Priority Candidates", len(high_priority))
                    
                    if len(high_priority) > 0:
                        fig_priority_esi = px.scatter(
                            high_priority,
                            x='earth_similarity_index',
                            y='research_priority',
                            size='koi_prad_original',
                            color='gravity_ratio',
                            hover_data=['kepid'],
                            title="ESI vs Research Priority (High Priority Candidates)",
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_priority_esi, use_container_width=True)
                
                with col2:
                    if 'research_priority' in candidate_df.columns:
                        priority_stats = candidate_df['research_priority'].describe()
                        st.write("Research Priority Statistics:")
                        st.dataframe(priority_stats)
        
        st.markdown("</div>", unsafe_allow_html=True)

    with tab6:
        st.header("Data Export & Summary")
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Downloads")
            csv_full = df.to_csv(index=False)
            st.download_button(
                "Download Full Dataset",
                data=csv_full,
                file_name="nasa_exoplanet_research_dataset.csv",
                mime="text/csv"
            )
            
            if not candidate_df.empty and 'earth_similarity_index' in candidate_df.columns:
                top_esi_candidates = candidate_df.nlargest(50, 'earth_similarity_index')
                csv_candidates = top_esi_candidates.to_csv(index=False)
                st.download_button(
                    "Download Top 50 Candidates by ESI",
                    data=csv_candidates,
                    file_name="top_esi_candidates.csv",
                    mime="text/csv"
                )
            
            if 'earth_similarity_index' in df.columns:
                esi_candidates = df[df['earth_similarity_index'] > 0.6].sort_values('earth_similarity_index', ascending=False)
                if len(esi_candidates) > 0:
                    csv_esi = esi_candidates.to_csv(index=False)
                    st.download_button(
                        "Download Earth-Like Candidates (ESI > 0.6)",
                        data=csv_esi,
                        file_name="earth_like_candidates.csv",
                        mime="text/csv"
                    )
                
                earth_gravity_candidates = df[(df['earth_similarity_index'] > 0.8) & 
                                            (df['gravity_ratio'] >= 0.8) & 
                                            (df['gravity_ratio'] <= 1.2)]
                if len(earth_gravity_candidates) > 0:
                    csv_earth_gravity = earth_gravity_candidates.to_csv(index=False)
                    st.download_button(
                        "Download True Earth Analogs",
                        data=csv_earth_gravity,
                        file_name="true_earth_analogs.csv",
                        mime="text/csv"
                    )
                
                if 'research_priority' in df.columns:
                    high_priority_candidates = df[df['research_priority'] > 0.7].sort_values('research_priority', ascending=False)
                    if len(high_priority_candidates) > 0:
                        csv_priority = high_priority_candidates.to_csv(index=False)
                        st.download_button(
                            "Download High Research Priority Candidates",
                            data=csv_priority,
                            file_name="high_priority_candidates.csv",
                            mime="text/csv"
                        )
        
        with col2:
            st.subheader("Dataset Summary")
            st.metric("Total Systems Analyzed", len(df))
            st.metric("Candidate Systems", len(candidate_df))
            st.metric("Available Features", len(features))
            if model_trained:
                st.metric("Model Performance", f"{cv_scores.mean():.3f}")
            
            if 'earth_similarity_index' in df.columns:
                earth_like_count = len(df[df['earth_similarity_index'] > 0.8])
                high_similarity_count = len(df[df['earth_similarity_index'] > 0.6])
                earth_gravity_count = len(df[(df['gravity_ratio'] >= 0.8) & (df['gravity_ratio'] <= 1.2)])
                high_priority_count = len(df[df['research_priority'] > 0.7]) if 'research_priority' in df.columns else 0
                
                st.metric("Earth-like Candidates (ESI > 0.8)", earth_like_count)
                st.metric("Highly Similar Candidates (ESI > 0.6)", high_similarity_count)
                st.metric("Earth-like Gravity Candidates", earth_gravity_count)
                st.metric("High Research Priority", high_priority_count)
        
        st.subheader("Enhanced Earth Similarity Index Summary")
        if 'earth_similarity_index' in df.columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average ESI", f"{df['earth_similarity_index'].mean():.3f}")
            with col2:
                st.metric("Maximum ESI", f"{df['earth_similarity_index'].max():.3f}")
            with col3:
                st.metric("Median ESI", f"{df['earth_similarity_index'].median():.3f}")
            with col4:
                st.metric("Average Gravity", f"{df['gravity_ratio'].mean():.2f}g")
            
            fig_esi_enhanced = px.histogram(
                df,
                x='earth_similarity_index',
                color='gravity_category' if 'gravity_category' in df.columns else None,
                nbins=20,
                title="Earth Similarity Index Distribution by Gravity Category",
                color_discrete_sequence=['#51cf66', '#fcc419', '#ff6b6b', '#ff922b']
            )
            st.plotly_chart(fig_esi_enhanced, use_container_width=True)
            
            st.subheader("ESI Component Analysis")
            component_stats = pd.DataFrame({
                'Component': ['Radius Similarity', 'Temperature Similarity', 'Flux Similarity', 'Gravity Similarity'],
                'Average Score': [
                    df['radius_esi'].mean() if 'radius_esi' in df.columns else 0,
                    df['temp_esi'].mean() if 'temp_esi' in df.columns else 0,
                    df['flux_esi'].mean() if 'flux_esi' in df.columns else 0,
                    df['gravity_esi'].mean() if 'gravity_esi' in df.columns else 0
                ]
            })
            
            fig_components = px.bar(component_stats, x='Component', y='Average Score',
                                  title="Average ESI Component Scores",
                                  color='Average Score',
                                  color_continuous_scale='Viridis')
            st.plotly_chart(fig_components, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    import traceback
    st.error(f"Detailed error: {traceback.format_exc()}")
    st.info("Please ensure your CSV file follows the expected Kepler data format")

st.sidebar.markdown("---")
st.sidebar.markdown("NASA Exoplanet Research Platform")
st.sidebar.markdown("Advanced machine learning for planetary candidate analysis & Earth similarity assessment")