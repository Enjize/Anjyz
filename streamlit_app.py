# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
from openai import OpenAI
import folium
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
from streamlit_folium import st_folium # For interactive maps in Streamlit
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Anjyz Analyzer | Investment Analysis", # Updated Title
    page_icon="💡", # Changed Icon
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar for potential future use or info
)

# --- Constants and File Paths ---
NEIGHBORHOOD_ID_TO_ANALYZE = "SA-RIY-YAS" # Assuming this remains the same
LAT_MIN, LAT_MAX = 24.79, 24.81
LON_MIN, LON_MAX = 46.62, 46.65
DEFAULT_MAP_CENTER = [24.80, 46.635] # Default center for Al Yasmin neighborhood
DEFAULT_MAP_ZOOM = 13
PROFILE_FILE = 'yasmen.json'
LICENSES_FILE = 'fake_licenses_SA-RIY-YAS_openai.json' # Keeping filename for now
POIS_FILE = 'fake_pois_SA-RIY-YAS_openai.json'      # Keeping filename for now

# --- Load API Key ---
client = None
try:
    # For Hugging Face Spaces / Streamlit Community Cloud, use st.secrets
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API key not found in Streamlit Secrets (OPENAI_API_KEY). Analysis functions might not work.", icon="⚠️")
    else:
        client = OpenAI(api_key=api_key)
        print("Streamlit App: OpenAI Client initialized successfully.")
except Exception as e:
    # Fallback for local testing if .streamlit/secrets.toml doesn't exist
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            print("Streamlit App: OpenAI Client initialized locally via .env.")
        else:
             st.error("Fatal Error: OpenAI API key not found in Secrets or .env. Please set it up.", icon="🚨")
    except ImportError:
         st.error("To run locally with an API key, please install python-dotenv and create a .env file.", icon="🚨")
    except Exception as inner_e:
        st.error(f"Failed to initialize OpenAI Client: {inner_e}", icon="🚨")


# --- Data Loading with Caching ---
@st.cache_data # Cache data to avoid reloading on every interaction
def load_data(profile_file, licenses_file, pois_file):
    try:
        with open(profile_file, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        print(f"Streamlit App: Data loaded successfully from {profile_file}")
    except Exception as e:
        st.error(f"Failed to load neighborhood profile '{profile_file}': {e}", icon="❌")
        success = False

    try:
        script_dir = os.path.dirname(__file__)
        licenses_path = os.path.join(script_dir, licenses_file)
        with open(licenses_path, 'r', encoding='utf-8') as f:
            licenses = json.load(f)
        print(f"Streamlit App: Data loaded successfully from {licenses_file}")
    except Exception as e:
        st.error(f"Failed to load licenses file '{licenses_file}': {e}", icon="❌")
        success = False
        
    try:
        script_dir = os.path.dirname(__file__)
        pois_path = os.path.join(script_dir, pois_file)
        with open(pois_path, 'r', encoding='utf-8') as f:
            pois = json.load(f)
        return profile, licenses, pois, True
    except Exception as e:
        st.error(f"Failed to load Points of Interest file '{pois_file}': {e}", icon="❌")
        # Non-critical, app might still work partially
        pois = [] # Default to empty list

    return profile, licenses, pois, success

neighborhood_profile, licenses_data, pois_data, data_loaded_successfully = load_data(PROFILE_FILE, LICENSES_FILE, POIS_FILE)

# --- Prepare Activity List for Dropdown (run only once after loading) ---
# *Updated to use English activity descriptions*
if 'available_activities_list' not in st.session_state and licenses_data:
    activities = []
    if isinstance(licenses_data, list):
        seen_descs = set()
        for lic in licenses_data:
            # Use English description for selection
            desc_en = lic.get('activity_description_en') 
            if desc_en and desc_en not in seen_descs:
                activities.append(desc_en)
                seen_descs.add(desc_en)
    activities.sort()
    st.session_state.available_activities_list = activities
elif 'available_activities_list' not in st.session_state:
    st.session_state.available_activities_list = []


# --- Helper Functions ---
def prepare_enhanced_context_for_llm(profile, licenses, pois, selected_activity_en, prop_lat, prop_lon):
    """Prepares a context string in English for the LLM."""
    if not all([profile, licenses, pois, selected_activity_en, prop_lat, prop_lon]): return None, None, []
    
    # Get profile data using English keys where available, fallback to Arabic or N/A
    neighborhood_name_en = profile.get('neighborhood_name_en','N/A')
    population = profile.get('demographics',{}).get('total_population','N/A')
    age_dist = profile.get('demographics',{}).get('age_distribution',{})
    youth_ratio_18_35 = age_dist.get('18-35',0.0)
    income = profile.get('socio_economic',{}).get('estimated_avg_monthly_income_sar','N/A')
    proposed_location = (prop_lat,prop_lon)
    
    competitors = []
    MAX_COMPETITORS_IN_CONTEXT = 5
    
    # Find competitors based on the selected English activity description
    for lic in licenses:
        if (lic.get('location',{}).get('neighborhood_id')==NEIGHBORHOOD_ID_TO_ANALYZE and 
            lic.get('activity_description_en')==selected_activity_en and
            lic.get('license_status', '').lower() == 'active'): # Consider only active licenses
            
            loc = lic.get('location',{})
            comp_lat=loc.get('latitude')
            comp_lon=loc.get('longitude')
            if comp_lat and comp_lon:
                try: distance = geodesic(proposed_location,(comp_lat,comp_lon)).km
                except: distance = None
                # Store competitor info using English names
                competitors.append({
                    "name": lic.get('business_name_en', lic.get('business_name_ar', 'N/A')), # Fallback to Arabic name if English missing
                    "lat": comp_lat,
                    "lon": comp_lon,
                    "distance_km": round(distance, 2) if distance is not None else None
                })

    competitor_count = len(competitors)
    competitors.sort(key=lambda x: x.get('distance_km') if x.get('distance_km') is not None else float('inf'))
    
    nearest_comp_info = "No active competitors found for this activity."
    comp_loc_list_str = "[]"
    if competitors:
        if competitors[0].get('distance_km') is not None:
            nearest_comp_info = f"Nearest competitor ~{competitors[0]['distance_km']:.2f} km away."
        # Create JSON string of competitor locations
        comp_loc_list_str = json.dumps(
            [{'lat': c['lat'], 'lon': c['lon'], 'dist_km': c['distance_km']} for c in competitors[:MAX_COMPETITORS_IN_CONTEXT]], 
            ensure_ascii=False
        )

    poi_summary = {}
    important_cats = ['Education', 'Healthcare', 'Shopping', 'Recreation', 'Religious', 'Services', 'Transportation', 'Government'] # Added more
    nearby_pois = []
    MAX_NEARBY_KM = 1.0
    if pois:
        for poi in pois:
            cat = poi.get('category', 'N/A')
            poi_summary[cat] = poi_summary.get(cat, 0) + 1
            if cat in important_cats:
                loc = poi.get('location', {})
                poi_lat = loc.get('latitude')
                poi_lon = loc.get('longitude')
                if poi_lat and poi_lon:
                    try:
                        dist = geodesic(proposed_location, (poi_lat, poi_lon)).km
                        if dist <= MAX_NEARBY_KM:
                            # Use English names for POIs
                            nearby_pois.append({
                                "name": poi.get('name_en', poi.get('name_ar', 'N/A')), # Fallback to Arabic
                                "cat": cat,
                                "dist_km": round(dist, 2)
                            })
                    except: 
                        pass # Ignore POIs if distance calculation fails

    nearby_pois.sort(key=lambda x: x['dist_km'])
    
    poi_context = "POI Summary: " + ", ".join([f"{c}: {n}" for c, n in poi_summary.items()]) if poi_summary else "No POI data available."
    nearby_poi_context = f"Nearby POIs (within {MAX_NEARBY_KM} km): "
    if nearby_pois:
        nearby_poi_context += ", ".join([f"{p['name']} ({p['cat']}) - {p['dist_km']:.2f}km" for p in nearby_pois[:5]]) # Show top 5 nearby
    else:
        nearby_poi_context += "None found."

    # Construct the final context string in English
    context = f"""Neighborhood Summary for {neighborhood_name_en}: Analysis for activity '{selected_activity_en}' at proposed location ({prop_lat:.5f}, {prop_lon:.5f}).
- Demographics: Population ~{population}, Youth (18-35) ratio ~{youth_ratio_18_35:.1%}, Avg Monthly Income ~{income} SAR.
- Competition: {competitor_count} active competitors found. {nearest_comp_info}. Locations of nearest {MAX_COMPETITORS_IN_CONTEXT}: {comp_loc_list_str}.
- Surroundings: {poi_context}. {nearby_poi_context}."""
    
    print("Streamlit App: English context prepared.")
    return context, proposed_location, competitors # Return competitors for map

def get_anjyz_analysis_enhanced(llm_client, context_summary):
    """Gets investment analysis from LLM in English."""
    if not llm_client: return "Error: OpenAI Client is not initialized."
    if not context_summary: return "Error: Data context is empty."
    
    # Updated prompt requesting English output and professional tone
    analysis_prompt = f"""You are an expert investment analyst for 'Anjyz', specializing in commercial opportunities in Saudi Arabia. Based on the following data summary, provide a clear investment recommendation from these options: 'Promising Opportunity', 'Viable with Considerations', or 'Saturated/Suboptimal Location'. Justify your recommendation by linking market competition, proposed location, neighborhood demographics/economics, and nearby points of interest. Respond in English with a professional and concise tone. Data: {context_summary}"""
    
    try:
        print("Streamlit App: Sending request to LLM...")
        completion = llm_client.chat.completions.create(
            model="o4-mini", 
            messages=[ 
                {"role": "system", "content": "You are a professional investment analyst."}, 
                {"role": "user", "content": analysis_prompt} 
            ] 
        )
        analysis_result = completion.choices[0].message.content
        print("Streamlit App: Received response from LLM.")
        return analysis_result
    except Exception as e:
        print(f"Streamlit App Error: LLM API call failed - {e}")
        st.error(f"An error occurred while contacting the language model: {e}", icon="🌐")
        return f"Apologies, an error occurred during data analysis."


def create_integrated_output_map(profile, pois, competitors_list, proposed_loc, selected_activity_en):
    """Creates the final Folium map object in English."""
    if not profile or not proposed_loc: return None
    center_lat, center_lon = proposed_loc[0], proposed_loc[1]
    zoom_start = 15
    output_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles='CartoDB positron')
    
    # Add Proposed Location marker (English)
    folium.Marker(
        location=proposed_loc, 
        popup=f"<b>Proposed Location</b><br>{selected_activity_en}", 
        tooltip="Proposed Location", 
        icon=folium.Icon(color='green', icon='star', prefix='fa')
    ).add_to(output_map)
    
    # Add Competitors (English)
    if competitors_list:
        comp_cluster = MarkerCluster(name=f"Competitors ({selected_activity_en})", overlay=True, control=True).add_to(output_map) 
        for comp in competitors_list:
            dist_text = f"{comp['distance_km']:.2f} km" if comp.get('distance_km') is not None else "N/A"
            # Use English name from competitor dict
            popup_html = f"<b>{comp['name']}</b><br><i>Competitor</i><br>Distance: {dist_text}"
            folium.Marker(
                location=[comp['lat'], comp['lon']], 
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{comp['name']} (Competitor)", 
                icon=folium.Icon(color='red', icon='briefcase', prefix='fa')
            ).add_to(comp_cluster)
            
    # Add POIs (English)
    if pois:
        poi_cluster = MarkerCluster(name="Points of Interest", overlay=True, control=True).add_to(output_map)
        for poi in pois:
           try:
                loc = poi.get('location', {})
                lat = loc.get('latitude'); lon = loc.get('longitude')
                if lat is not None and lon is not None:
                        # Use English POI names and categories
                        name = poi.get('name_en', poi.get('name_ar', 'N/A'))
                        category = poi.get('category','N/A'); 
                        subcategory = poi.get('subcategory','')
                        icon_name = 'info-circle'; icon_color = 'blue'
                        # Simplified icon logic based on English categories
                        cat_lower = category.lower()
                        if 'education' in cat_lower: icon_name='graduation-cap'; icon_color='darkblue'
                        elif 'health' in cat_lower: icon_name='hospital-o'; icon_color='darkred' # Adjusted icon color
                        elif 'shopping' in cat_lower: icon_name='shopping-cart'; icon_color='purple'
                        elif 'recreation' in cat_lower: icon_name='tree'; icon_color='darkgreen' # Adjusted icon color
                        elif 'religious' in cat_lower: icon_name='moon-o'; icon_color='orange'   # Adjusted icon color
                        elif 'services' in cat_lower: icon_name='bank'; icon_color='cadetblue'
                        elif 'transportation' in cat_lower: icon_name='bus'; icon_color='black'
                        elif 'government' in cat_lower: icon_name='building'; icon_color='gray'

                        popup_html = f"<b>{name}</b><br>{category}"
                        if subcategory: popup_html += f" ({subcategory})"
                        
                        folium.Marker(
                            location=[lat, lon], 
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=name, 
                            icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa', icon_size=(20,20)) # Smaller icons
                        ).add_to(poi_cluster)
           except Exception as e: 
               print(f"Map Error adding POI {poi.get('poi_id', '')}: {e}")
               
    folium.LayerControl().add_to(output_map) # Add layer control to toggle POIs/Competitors
    print("Streamlit App: English output map object created.")
    return output_map # Return the Folium map object

# --- Streamlit App UI ---

st.title("💡 Anjyz Analyzer")
st.markdown("Analyze commercial investment opportunities based on location and activity.")
st.divider()

# --- Input Section ---
st.subheader("1. Define Your Proposal")
input_col1, input_col2 = st.columns([2, 1]) # Give map more space (2/3rds width)

with input_col1:
    st.markdown("**Specify Proposed Location (Click on Map):**")

    # --- Custom CSS for Map Container ---
    # Note: The specific class name might change with Streamlit versions or if nested differently.
    # Inspect the element in your browser's developer tools if styling doesn't apply.
    st.markdown("""
    <style>
    /* Target the iframe container within Streamlit columns more reliably */
    .stDataFrame iframe {
        border: 1px solid #e0e0e0; /* Light grey border */
        border-radius: 8px;      /* Rounded corners */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Subtle shadow */
    }
    /* Optional: Adjust container padding if needed */
    .stDataFrame {
         padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize map centered on Yasmin
    input_map = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM, tiles="CartoDB positron")
    # Add polygon for context (optional) - MODIFIED FOR FILL
    try:
        poly_coords = neighborhood_profile['geometry']['coordinates'][0]
        folium_poly_coords = [(coord[1], coord[0]) for coord in poly_coords]
        folium.Polygon(
            locations=folium_poly_coords,
            # Outline settings (optional, can be removed or made subtle)
            color='#007bff',  # A subtle blue outline
            weight=1,         # Thin outline
            # Fill settings
            fill=True,
            fill_color='#007bff', # Same blue color for fill
            fill_opacity=0.15,    # Adjust transparency (0.0 to 1.0)
            tooltip=f"{neighborhood_profile.get('neighborhood_name_en', 'Neighborhood')} Boundary" # Updated tooltip
            # Removed dash_array
        ).add_to(input_map)
        print("Streamlit App: Added neighborhood boundary fill to input map.")
    except Exception as e:
        print(f"Streamlit Warning: Could not draw neighborhood polygon on input map - {e}")

    # Use st_folium for interactive map input
    if 'map_data' not in st.session_state:
        st.session_state['map_data'] = {'last_clicked': None}

    # Add a marker if a location is already selected
    if st.session_state['map_data'].get('last_clicked'):
        click_coords = (st.session_state['map_data']['last_clicked']['lat'], st.session_state['map_data']['last_clicked']['lng'])
        folium.Marker(
            location=click_coords,
            tooltip="Selected Location",
            icon=folium.Icon(color='blue', icon='map-marker', prefix='fa')
        ).add_to(input_map)


    map_interaction = st_folium(
        input_map,
        key="input_map_interaction_main", # New key for main area map
        height=400, # Adjust height as needed
        width=700, # Adjust width as needed
        returned_objects=['last_clicked']
    )

    # Store clicked coordinates if interaction happened
    if map_interaction and map_interaction.get('last_clicked'):
        # Check if click is different from previous to avoid unnecessary reruns if state handling allows
        if st.session_state['map_data'].get('last_clicked') != map_interaction.get('last_clicked'):
            st.session_state['map_data'] = map_interaction
            # Force rerun to update the marker and selected coordinates display immediately
            st.rerun()

with input_col2:
    # Display selected coordinates
    prop_lat = None
    prop_lon = None
    if st.session_state['map_data'] and st.session_state['map_data'].get('last_clicked'):
        clicked_data = st.session_state['map_data']['last_clicked']
        prop_lat = clicked_data['lat']
        prop_lon = clicked_data['lng'] # st_folium uses 'lng'
        st.success(f"**📍 Location Selected:**\nLat: {prop_lat:.6f}\nLon: {prop_lon:.6f}", icon="✅")
    else:
        st.info("👆 Click the map to select the proposed location.", icon="🗺️")

    st.markdown("**Select Business Activity:**")
    selected_activity_en = st.selectbox(
        "Select Business Activity:", # Label can be hidden if title is enough
        label_visibility="collapsed", # Hide label, use markdown title above
        options=st.session_state.get('available_activities_list', ["Loading..."]),
        index=None, # No default selection
        placeholder="Choose an activity..."
    )

    st.markdown("---") # Separator before button

    # Analysis Button
    analyze_button = st.button("🚀 Analyze Opportunity", type="primary", use_container_width=True, disabled=(not selected_activity_en or prop_lat is None))

st.divider() # Divider between input and output sections

# --- Results Section ---
st.subheader("2. Analysis Results")

if analyze_button:
    # Check all prerequisites again (safety check)
    if client and data_loaded_successfully and selected_activity_en and prop_lat is not None and prop_lon is not None:
        st.info(f"Analyzing opportunity for '{selected_activity_en}' at ({prop_lat:.5f}, {prop_lon:.5f})...", icon="⏳")

        with st.spinner("Preparing context and generating analysis... Please wait."):
            # 1. Prepare Context
            analysis_context, proposed_location_tuple, competitors_list = prepare_enhanced_context_for_llm(
                neighborhood_profile, licenses_data, pois_data,
                selected_activity_en, prop_lat, prop_lon
            )

            # 2. Get LLM Analysis
            analysis_text = "Failed to retrieve analysis."
            if analysis_context:
                analysis_text = get_anjyz_analysis_enhanced(client, analysis_context)

            # 3. Create Output Map
            output_map_object = None
            if proposed_location_tuple:
                 output_map_object = create_integrated_output_map(
                     neighborhood_profile, pois_data, competitors_list,
                     proposed_location_tuple, selected_activity_en
                 )

        # 4. Display Results
        st.markdown("**Recommendation & Analysis:**")
        st.markdown(analysis_text) # Display LLM analysis

        st.markdown("---")

        st.markdown("**Interactive Results Map:**")
        if output_map_object:
            st_folium(output_map_object,
                      key="output_map_display", # Different key
                      height=500,
                      width=700, # Match width potentially
                      returned_objects=[]) # No return needed
        else:
            st.warning("Could not generate the results map.", icon="🗺️")

    else:
        # Handle missing prerequisites before analysis (more user-friendly messages)
        if prop_lat is None or prop_lon is None: st.error("❌ Please select a location on the map first.", icon="📍")
        if not selected_activity_en: st.error("❌ Please select a business activity.", icon="📋")
        if not client: st.error("🔑 OpenAI client initialization error. Check API key setup.", icon="🔑")
        if not data_loaded_successfully: st.error("💾 Failed to load essential data. Analysis cannot proceed.", icon="💾")

else:
    # Initial message when no analysis has been run yet
    st.info("Define your proposal above and click 'Analyze Opportunity' to see the results here.")


# --- Sidebar (Optional: Keep for less critical info or future use) ---
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("This tool provides investment analysis based on location data and AI.")
    # Add any other info, links, or less critical controls here
    st.markdown("---")
    # Removed mention of fake data
    st.markdown("___")