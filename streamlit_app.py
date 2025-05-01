# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
from openai import OpenAI
import folium
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
from streamlit_folium import st_folium
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Anjez Analyzer | Anjez Analysis",
    page_icon="😺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
NEIGHBORHOOD_ID_TO_ANALYZE = "SA-RIY-YAS"
DEFAULT_MAP_CENTER = [24.80, 46.635]
DEFAULT_MAP_ZOOM = 13
PROFILE_FILE = 'yasmen.json'
LICENSES_FILE = 'fake_licenses_SA-RIY-YAS_openai.json'
POIS_FILE = 'fake_pois_SA-RIY-YAS_openai.json'

# --- Load API Key ---
client = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
except:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
    except:
        pass

# --- Load Data ---
@st.cache_data
def load_data(profile_file, licenses_file, pois_file):
    try:
        with open(profile_file, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        with open(licenses_file, 'r', encoding='utf-8') as f:
            licenses = json.load(f)
        with open(pois_file, 'r', encoding='utf-8') as f:
            pois = json.load(f)
        return profile, licenses, pois, True
    except Exception as e:
        st.error(f"Failed to load data: {e}", icon="❌")
        return None, None, None, False

neighborhood_profile, licenses_data, pois_data, data_loaded_successfully = load_data(PROFILE_FILE, LICENSES_FILE, POIS_FILE)

# --- Prepare Activity List for Dropdown ---
if 'available_activities_list' not in st.session_state and licenses_data:
    activities = []
    if isinstance(licenses_data, list):
        seen_descs = set()
        for lic in licenses_data:
            desc = lic.get('activity_description_en')
            if desc and desc not in seen_descs:
                activities.append(desc)
                seen_descs.add(desc)
    activities.sort()
    st.session_state.available_activities_list = activities
elif 'available_activities_list' not in st.session_state:
    st.session_state.available_activities_list = []

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/MohammadAlzhrani/Athar/main/athar-high-resolution-logo-transparent.png", width=150)
    st.title("📍 Anjez Input Panel")
    st.markdown("Choose a business type and select the proposed location on the map.")

    selected_activity = st.selectbox(
        "1. Select Business Activity:",
        options=st.session_state.available_activities_list,
        index=None,
        placeholder="Select an activity..."
    )

    st.markdown("2. Click the map to select a proposed location:")
    input_map = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM, tiles="CartoDB positron")
    try:
        poly_coords = neighborhood_profile['geometry']['coordinates'][0]
        folium_poly_coords = [(coord[1], coord[0]) for coord in poly_coords]
        folium.Polygon(locations=folium_poly_coords, color='grey', fill=False, weight=1).add_to(input_map)
    except: pass

    if 'map_data' not in st.session_state:
        st.session_state['map_data'] = {'last_clicked': None}

    map_interaction = st_folium(input_map, center=DEFAULT_MAP_CENTER, zoom=DEFAULT_MAP_ZOOM, key="input_map_interaction", height=300, width=700, returned_objects=['last_clicked'])

    if map_interaction and map_interaction.get('last_clicked'):
        st.session_state['map_data'] = map_interaction

    prop_lat = prop_lon = None
    if st.session_state['map_data'].get('last_clicked'):
        clicked_data = st.session_state['map_data']['last_clicked']
        prop_lat = clicked_data['lat']
        prop_lon = clicked_data['lng']
        st.info(f"📍 Selected Location: ({prop_lat:.6f}, {prop_lon:.6f})", icon="✅")
    else:
        st.warning("Please click on the map to select a location.", icon="👆")

    analyze_button = st.button("🚀 Analyze Opportunity", type="primary", disabled=(not selected_activity or prop_lat is None))

# --- Main UI ---
st.title("📊 Anjez Analysis Results")

def get_anjez_analysis(llm_client, context_summary):
    if not llm_client or not context_summary:
        return "Unable to analyze data."
    analysis_prompt = f"You are an intelligent assistant in the Anjez project, helping analyze commercial investment opportunities in Saudi Arabia. Data: {context_summary}"
    try:
        completion = llm_client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "Investment analysis assistant."},
                {"role": "user", "content": analysis_prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Model call failed: {e}", icon="🌐")
        return "An error occurred during analysis."

if analyze_button:
    if client and data_loaded_successfully and selected_activity and prop_lat and prop_lon:
        st.info(f"Analyzing '{selected_activity}' at location ({prop_lat:.5f}, {prop_lon:.5f})...", icon="⏳")
        with st.spinner("Analyzing..."):
            context = f"Analysis for activity '{selected_activity}' in Al Yasmin neighborhood at coordinates ({prop_lat}, {prop_lon})."
            result = get_anjez_analysis(client, context)
            st.subheader("Recommendation and Analysis:")
            st.markdown(result)
    else:
        st.error("Please check the inputs before running the analysis.", icon="❗")

st.sidebar.markdown("---")
st.sidebar.markdown("*Demo data used for illustrative purposes only.*")
