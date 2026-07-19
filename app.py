import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import date

# --- 1. Page Configuration & Caching ---
st.set_page_config(page_title="Pokémon Type Predictor", layout="wide")

@st.cache_resource
def load_models():
    try:
        model = joblib.load("pokemon_classifier.pkl")
        artifacts = joblib.load("pokemon_artifacts.pkl")
        return model, artifacts
    except FileNotFoundError:
        return None, None

@st.cache_data
def get_data(_artifacts):
    return _artifacts["raw_df"], _artifacts["cleaned_df"]

model, artifacts = load_models()

# --- 2. Sidebar Navigation & Metadata ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["1. Intro & Overview", "2. Data Preview", "3. EDA & Visuals", "4. Model Performance", "5. Live Prediction"])

st.sidebar.markdown("---")
st.sidebar.markdown("**VISHAL RAMESH**")
st.sidebar.markdown("*2509_A_P4DSBI_P_FGA 8E076A*")
st.sidebar.markdown("*Python for Data Science*")
st.sidebar.markdown(f"*Last Updated: {date.today().strftime('%B %d, %Y')}*")
st.sidebar.caption("[Source: PokeAPI](https://pokeapi.co/)")

if not model or not artifacts:
    st.error("⚠️ Required model files not found. Please run `python train_model.py` first.")
    st.stop()

raw_df, cleaned_df = get_data(artifacts)

# --- SECTION 1: Intro & Overview ---
if page == "1. Intro & Overview":
    st.title("⚡ Pokémon Type Predictor")
    st.markdown("This dashboard leverages base combat statistics to predict a Pokémon's primary elemental type. As a complex multi-class problem with overlapping stat distributions, it serves as an end-to-end demonstration of data engineering and machine learning.")
    st.markdown("[🔗 View Live Data Source (PokeAPI)](https://pokeapi.co/)")
    
    st.subheader("Dashboard Headline Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Raw Records Mined", len(raw_df))
    col2.metric("Target Elements (Classes)", len(artifacts["classes"]))
    col3.metric("Selected Model", artifacts["metrics_df"].iloc[0]["Model"], "Hard Multi-Class")

# --- SECTION 2: Data Overview ---
elif page == "2. Data Preview":
    st.title("🗄️ Data Engineering Overview")
    
    st.markdown("### Pre vs. Post Cleaning Statistics")
    col1, col2 = st.columns(2)
    missing_raw = raw_df.isna().sum().sum()
    missing_clean = cleaned_df.isna().sum().sum()
    
    col1.metric("Raw DataFrame Shape", f"{raw_df.shape[0]} rows, {raw_df.shape[1]} cols", f"{missing_raw} missing values", delta_color="inverse")
    col2.metric("Cleaned DataFrame Shape", f"{cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} cols", f"{missing_clean} missing values")
    
    tab1, tab2 = st.tabs(["Raw JSON-Extracted Data", "Cleaned & Engineered Data"])
    with tab1:
        st.dataframe(raw_df.head(100), use_container_width=True)
    with tab2:
        st.dataframe(cleaned_df.head(100), use_container_width=True)

# --- SECTION 3: Exploratory Data Analysis ---
elif page == "3. EDA & Visuals":
    st.title("📊 Exploratory Data Analysis")
    
    target_filter = st.sidebar.multiselect("Filter Elemental Types:", options=artifacts["classes"], default=artifacts["classes"])
    filtered_df = cleaned_df[cleaned_df['type'].isin(target_filter)]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Class Balance Histogram**")
        fig1 = px.histogram(filtered_df, x="type", color="type", title="Distribution of Pokémon Types")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.markdown("**Stat Correlation (Defense vs Attack)**")
        fig2 = px.scatter(filtered_df, x="defense", y="attack", color="type", size="total_stats", opacity=0.7, title="Combat Stat Clustering")
        st.plotly_chart(fig2, use_container_width=True)

# --- SECTION 4: Model Performance ---
elif page == "4. Model Performance":
    st.title("🤖 Model Evaluation Metrics")
    st.info("Note: Predicting type solely from combat stats is a difficult multi-class problem. Modest accuracy is expected as many types share identical statistical boundaries.")
    
    st.markdown("### Comparative Performance")
    st.dataframe(artifacts["metrics_df"], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion Matrix (Random Forest)")
        fig_cm = px.imshow(artifacts["cm"], text_auto=True, color_continuous_scale='Reds', 
                           x=artifacts["classes"], y=artifacts["classes"],
                           labels=dict(x="Predicted Type", y="Actual Type", color="Count"))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        st.markdown("### Feature Importance")
        feat_df = pd.DataFrame({
            "Feature": artifacts["feature_names"],
            "Importance": artifacts["feature_importances"]
        }).sort_values(by="Importance", ascending=True)
        fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation='h')
        st.plotly_chart(fig_feat, use_container_width=True)

# --- SECTION 5: Live Prediction ---
elif page == "5. Live Prediction":
    st.title("🎯 Elemental Prediction Engine")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Base Stats")
        with st.form("prediction_form"):
            hp = st.slider("HP", 10, 255, 45)
            attack = st.slider("Attack", 5, 190, 49)
            defense = st.slider("Defense", 5, 230, 49)
            sp_atk = st.slider("Special Attack", 10, 194, 65)
            sp_def = st.slider("Special Defense", 20, 230, 65)
            speed = st.slider("Speed", 5, 180, 45)
            
            # Auto-calculate the engineered feature
            total_stats = hp + attack + defense + sp_atk + sp_def + speed
            
            submitted = st.form_submit_button("Predict Type")
            
    with col2:
        if submitted:
            input_data = pd.DataFrame([[hp, attack, defense, sp_atk, sp_def, speed, total_stats]], columns=artifacts["feature_names"])
            
            prediction = model.predict(input_data)[0]
            st.success(f"### Predicted Primary Type: **{prediction.upper()}**")
            
            probs = model.predict_proba(input_data)[0]
            prob_df = pd.DataFrame({"Type": artifacts["classes"], "Probability": probs}).sort_values(by="Probability", ascending=True)
            
            fig_prob = px.bar(prob_df, x="Probability", y="Type", orientation='h', text_auto='.1%', title="Type Probability Distribution")
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("Adjust the combat stats on the left and click **Predict Type** to run the live algorithm.")
