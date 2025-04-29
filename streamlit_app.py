# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
from openai import OpenAI
import folium
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
from streamlit_folium import st_folium # <- لاستخدام الخرائط التفاعلية في Streamlit
import time

# --- Page Configuration (Set Title, Icon, Layout) ---
st.set_page_config(
    page_title="Athar Analyzer | تحليل أثر",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants and File Paths ---
NEIGHBORHOOD_ID_TO_ANALYZE = "SA-RIY-YAS"
LAT_MIN, LAT_MAX = 24.79, 24.81
LON_MIN, LON_MAX = 46.62, 46.65
DEFAULT_MAP_CENTER = [24.80, 46.635] # مركز افتراضي لحي الياسمين
DEFAULT_MAP_ZOOM = 13

PROFILE_FILE = 'yasmen.json'
LICENSES_FILE = 'fake_licenses_SA-RIY-YAS_openai.json'
POIS_FILE = 'fake_pois_SA-RIY-YAS_openai.json'

# --- Load API Key (Use Streamlit Secrets for Deployment) ---
client = None
try:
    # For Hugging Face Spaces / Streamlit Community Cloud, use st.secrets
    api_key = st.secrets.get("OPENAI_API_KEY") 
    if not api_key:
        st.warning("لم يتم العثور على مفتاح OpenAI API في Streamlit Secrets (OPENAI_API_KEY). قد لا تعمل وظائف التحليل.", icon="⚠️")
    else:
        client = OpenAI(api_key=api_key)
        print("Streamlit App: OpenAI Client initialized successfully.")
except Exception as e:
    # Fallback for local testing if .streamlit/secrets.toml doesn't exist
    # Requires a .env file in the root with OPENAI_API_KEY=...
    # You need to run: pip install python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            print("Streamlit App: OpenAI Client initialized locally via .env.")
        else:
             st.error("خطأ فادح: لم يتم العثور على مفتاح OpenAI API لا في Secrets ولا في .env. يرجى إعداده.", icon="🚨")
    except ImportError:
         st.error("لتشغيل التطبيق محليًا مع مفتاح API، يرجى تثبيت python-dotenv وإنشاء ملف .env", icon="🚨")
    except Exception as inner_e:
        st.error(f"فشل إعداد OpenAI Client: {inner_e}", icon="🚨")


# --- Data Loading with Caching ---
@st.cache_data # Cache data to avoid reloading on every interaction
def load_data(profile_file, licenses_file, pois_file):
    """Loads all necessary JSON data files."""
    profile = None
    licenses = None
    pois = None
    success = True
    try:
        script_dir = os.path.dirname(__file__)
        profile_path = os.path.join(script_dir, profile_file)
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        print(f"Streamlit App: Data loaded successfully from {profile_file}")
    except Exception as e:
        st.error(f"فشل تحميل ملف بيانات الحي '{profile_file}': {e}", icon="❌")
        success = False

    try:
        script_dir = os.path.dirname(__file__)
        licenses_path = os.path.join(script_dir, licenses_file)
        with open(licenses_path, 'r', encoding='utf-8') as f:
            licenses = json.load(f)
        print(f"Streamlit App: Data loaded successfully from {licenses_file}")
    except Exception as e:
        st.error(f"فشل تحميل ملف التراخيص '{licenses_file}': {e}", icon="❌")
        success = False
        
    try:
        script_dir = os.path.dirname(__file__)
        pois_path = os.path.join(script_dir, pois_file)
        with open(pois_path, 'r', encoding='utf-8') as f:
            pois = json.load(f)
        print(f"Streamlit App: Data loaded successfully from {pois_file}")
    except Exception as e:
        st.error(f"فشل تحميل ملف نقاط الاهتمام '{pois_file}': {e}", icon="❌")
        # Non-critical, app might still work partially
        pois = [] # Default to empty list

    return profile, licenses, pois, success

neighborhood_profile, licenses_data, pois_data, data_loaded_successfully = load_data(PROFILE_FILE, LICENSES_FILE, POIS_FILE)

# --- Prepare Activity List for Dropdown (run only once after loading) ---
if 'available_activities_list' not in st.session_state and licenses_data:
    activities = []
    if isinstance(licenses_data, list):
        seen_descs = set()
        for lic in licenses_data:
            desc = lic.get('activity_description_ar')
            if desc and desc not in seen_descs:
                activities.append(desc)
                seen_descs.add(desc)
    activities.sort()
    st.session_state.available_activities_list = activities
    print(f"Streamlit App: Found {len(activities)} unique activities.")
elif 'available_activities_list' not in st.session_state:
     st.session_state.available_activities_list = []


# --- Helper Functions (Context Prep, LLM Call, Map Creation - adapted for Streamlit) ---
# (These functions remain largely the same logic as before, just ensure they work with the loaded data)
def prepare_enhanced_context_for_llm(profile, licenses, pois, selected_activity, prop_lat, prop_lon):
    # ...(Implementation from previous responses)...
    # Returns: context_string, proposed_location_tuple, competitors_list
    if not all([profile, licenses, pois, selected_activity, prop_lat, prop_lon]): return None, None, []
    neighborhood_name = profile.get('neighborhood_name_ar','N/A'); population = profile.get('demographics',{}).get('total_population','N/A')
    age_dist = profile.get('demographics',{}).get('age_distribution',{}); youth_ratio_18_35 = age_dist.get('18-35',0.0)
    income = profile.get('socio_economic',{}).get('estimated_avg_monthly_income_sar','N/A'); proposed_location = (prop_lat,prop_lon)
    competitors = []; MAX_COMPETITORS_IN_CONTEXT = 5
    for lic in licenses:
        if lic.get('location',{}).get('neighborhood_id')==NEIGHBORHOOD_ID_TO_ANALYZE and lic.get('activity_description_ar')==selected_activity:
            loc = lic.get('location',{}); comp_lat=loc.get('latitude'); comp_lon=loc.get('longitude')
            if comp_lat and comp_lon:
                try: distance = geodesic(proposed_location,(comp_lat,comp_lon)).km
                except: distance = None
                competitors.append({"name":lic.get('business_name_ar','N/A'),"lat":comp_lat,"lon":comp_lon,"distance_km":round(distance,2) if distance is not None else None})
    competitor_count = len(competitors)
    competitors.sort(key=lambda x: x.get('distance_km') if x.get('distance_km') is not None else float('inf'))
    nearest_comp_info="لا يوجد منافسون." ; comp_loc_list_str="[]"
    if competitors:
        if competitors[0].get('distance_km') is not None: nearest_comp_info = f"أقرب منافس ~{competitors[0]['distance_km']:.2f} كم."
        comp_loc_list_str=json.dumps([{'lat':c['lat'],'lon':c['lon'],'dist_km':c['distance_km']} for c in competitors[:MAX_COMPETITORS_IN_CONTEXT]], ensure_ascii=False)

    poi_summary = {}; important_cats=['Education','Healthcare','Shopping','Recreation','Religious','Services']; nearby_pois=[]; MAX_NEARBY_KM=1.0
    if pois:
        for poi in pois:
            cat=poi.get('category','N/A'); poi_summary[cat]=poi_summary.get(cat,0)+1
            if cat in important_cats:
                loc=poi.get('location',{}); poi_lat=loc.get('latitude'); poi_lon=loc.get('longitude')
                if poi_lat and poi_lon:
                    try:
                        dist=geodesic(proposed_location,(poi_lat,poi_lon)).km
                        if dist<=MAX_NEARBY_KM: nearby_pois.append({"name":poi.get('name_ar','N/A'),"cat":cat,"dist_km":round(dist,2)})
                    except: pass
    nearby_pois.sort(key=lambda x: x['dist_km'])
    poi_context = "ملخص POIs: "+", ".join([f"{c}: {n}" for c,n in poi_summary.items()]) if poi_summary else "لا توجد بيانات POI."
    nearby_poi_context = f"POIs هامة قريبة (ضمن {MAX_NEARBY_KM} كم): "
    if nearby_pois: nearby_poi_context += ", ".join([f"{p['name']}({p['cat']})-{p['dist_km']:.2f}كم" for p in nearby_pois[:5]]) # Show top 5 nearby
    else: nearby_poi_context += "لا يوجد."

    context = f"""ملخص لـ {neighborhood_name}: نشاط '{selected_activity}' موقع ({prop_lat:.5f}, {prop_lon:.5f}).
    - سكان: {population}, شباب(18-35): {youth_ratio_18_35:.1%}, دخل شهري: {income} ريال.
    - منافسة: {competitor_count} منافسين. {nearest_comp_info}. مواقع أقرب {MAX_COMPETITORS_IN_CONTEXT}: {comp_loc_list_str}.
    - محيط: {poi_context}. {nearby_poi_context}."""
    print("Streamlit App: Context prepared.")
    return context, proposed_location, competitors # Return competitors for map

def get_athar_analysis_enhanced(llm_client, context_summary):
    # ...(Implementation from previous responses - ensure error handling)...
    if not llm_client: return "خطأ: OpenAI Client غير مهيأ."
    if not context_summary: return "خطأ: سياق البيانات فارغ."
    analysis_prompt = f"""أنت مساعد "أثر" لتحليل الاستثمار بالسعودية. حلل البيانات التالية وقدم توصية واضحة (فرصة واعدة, ممكن مع اعتبارات, منطقة مشبعة/موقع غير مثالي) مع تعليل موجز (100-150 كلمة) يربط بين المنافسة، الموقع المقترح، بيانات الحي، ونقاط الاهتمام القريبة. الرد بالعربية.\n\nالبيانات:\n{context_summary}"""
    try:
        print("Streamlit App: Sending request to LLM...")
        completion = llm_client.chat.completions.create(
            model="gpt-4o", messages=[ {"role": "system", "content": "مساعد تحليل استثماري متخصص."}, {"role": "user", "content": analysis_prompt} ],
            temperature=0.6, max_tokens=300 )
        analysis_result = completion.choices[0].message.content
        print("Streamlit App: Received response from LLM.")
        return analysis_result
    except Exception as e:
        print(f"Streamlit App Error: LLM API call failed - {e}")
        st.error(f"حدث خطأ أثناء التواصل مع النموذج اللغوي: {e}", icon="🌐")
        return f"عذرًا، حدث خطأ أثناء محاولة تحليل البيانات."


def create_integrated_output_map(profile, pois, competitors_list, proposed_loc, selected_activity):
    """Creates the final Folium map object (does not save to file)."""
    # ...(Implementation from previous responses - MUST RETURN MAP OBJECT)...
    if not profile or not proposed_loc: return None
    center_lat, center_lon = proposed_loc[0], proposed_loc[1]
    zoom_start = 15
    output_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles='CartoDB positron')
    # Add Proposed Loc marker
    folium.Marker(location=proposed_loc, popup=f"<b>الموقع المقترح</b><br>{selected_activity}", tooltip="الموقع المقترح", icon=folium.Icon(color='green', icon='star', prefix='fa')).add_to(output_map)
    # Add Competitors
    if competitors_list:
        comp_cluster = MarkerCluster(name=f"منافسون ({selected_activity})", overlay=False, control=False).add_to(output_map) # control=False to prevent separate layer control for cluster
        for comp in competitors_list:
            dist_text = f"{comp['distance_km']:.2f} كم" if comp.get('distance_km') is not None else "N/A"
            popup_html = f"<b>{comp['name']}</b><br><i>منافس</i><br>المسافة: {dist_text}"
            folium.Marker(location=[comp['lat'], comp['lon']], popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{comp['name']} (منافس)", icon=folium.Icon(color='red', icon='briefcase', prefix='fa')
            ).add_to(comp_cluster)
    # Add POIs
    if pois:
        poi_cluster = MarkerCluster(name="نقاط الاهتمام", overlay=False, control=False).add_to(output_map)
        for poi in pois:
           try:
                loc = poi.get('location', {})
                lat = loc.get('latitude'); lon = loc.get('longitude')
                if lat is not None and lon is not None:
                        name = poi.get('name_ar','N/A'); category = poi.get('category','N/A'); subcategory = poi.get('subcategory','')
                        icon_name = 'info-circle'; icon_color = 'blue'
                        # Simplified icon logic
                        if 'edu' in category.lower(): icon_name='graduation-cap'; icon_color='darkblue'
                        elif 'health' in category.lower(): icon_name='hospital-o'; icon_color='red'
                        elif 'shop' in category.lower(): icon_name='shopping-cart'; icon_color='purple'
                        elif 'recr' in category.lower(): icon_name='tree'; icon_color='green'
                        elif 'relig' in category.lower(): icon_name='moon-o'; icon_color='darkgreen'
                        elif 'servi' in category.lower(): icon_name='bank'; icon_color='cadetblue'
                        popup_html = f"<b>{name}</b><br>{category} ({subcategory})"
                        folium.Marker(location=[lat, lon], popup=folium.Popup(popup_html, max_width=300),
                            tooltip=name, icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa', icon_size=(20,20))
                        ).add_to(poi_cluster)
           except Exception as e: print(f"Map Error adding POI {poi.get('poi_id', '')}: {e}")
    # Don't add LayerControl if using clusters this way, keep map cleaner
    # folium.LayerControl().add_to(output_map)
    print("Streamlit App: Output map object created.")
    return output_map # Return the Folium map object

# --- Streamlit App UI ---

# --- Sidebar for Inputs ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/MohammadAlzhrani/Athar/main/athar-high-resolution-logo-transparent.png", width=150) # استبدل برابط شعار أثر إذا وجد
    st.title("📍 مدخلات تحليل أثر")
    st.markdown("اختر النشاط وحدد الموقع المقترح على الخريطة.")

    # 1. Activity Selection
    selected_activity = st.selectbox(
        "1. اختر النشاط التجاري:",
        options=st.session_state.get('available_activities_list', ["الرجاء الانتظار..."]),
        index=None, # No default selection
        placeholder="اختر نشاطًا..."
    )

    # 2. Location Input Map
    st.markdown("2. حدد الموقع المقترح بالنقر على الخريطة:")
    # Initialize map centered on Yasmin
    input_map = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM, tiles="CartoDB positron")
    # Add polygon for context (optional)
    try:
        poly_coords = neighborhood_profile['geometry']['coordinates'][0]
        folium_poly_coords = [(coord[1], coord[0]) for coord in poly_coords]
        folium.Polygon(locations=folium_poly_coords, color='grey', fill=False, weight=1, tooltip="حدود حي الياسمين (تقريبي)").add_to(input_map)
    except Exception as e:
        print(f"Streamlit Warning: Could not draw neighborhood polygon on input map - {e}")

    # Use st_folium for interactive map input
    # Set a default key to avoid issues on first load if nothing is clicked
    if 'map_data' not in st.session_state:
         st.session_state['map_data'] = {'last_clicked': None}

    map_interaction = st_folium(
        input_map,
        center=DEFAULT_MAP_CENTER,
        zoom=DEFAULT_MAP_ZOOM,
        key="input_map_interaction", # Unique key for this map instance
        height=300, # Adjust height
        width=700, # Adjust width
        returned_objects=['last_clicked'] # We only need the last click data
    )

    # Store clicked coordinates in session state
    if map_interaction and map_interaction.get('last_clicked'):
        st.session_state['map_data'] = map_interaction # Store the whole interaction data which includes last_clicked
        
    # Display selected coordinates for confirmation
    prop_lat = None
    prop_lon = None
    if st.session_state['map_data'] and st.session_state['map_data'].get('last_clicked'):
        clicked_data = st.session_state['map_data']['last_clicked']
        prop_lat = clicked_data['lat']
        prop_lon = clicked_data['lng'] # Note: st_folium uses 'lng' not 'lon'
        st.info(f"📍 الموقع المحدد: ({prop_lat:.6f}, {prop_lon:.6f})", icon="✅")
    else:
        st.warning("الرجاء النقر على الخريطة لتحديد الموقع المقترح.", icon="👆")


    # 3. Analysis Button
    analyze_button = st.button("🚀 تحليل الفرصة", type="primary", disabled=(not selected_activity or prop_lat is None))

# --- Main Area for Outputs ---
st.title("📊 نتائج تحليل أثر")

if analyze_button:
    if client and data_loaded_successfully and selected_activity and prop_lat and prop_lon:
        st.info(f"جاري تحليل فرصة '{selected_activity}' في الموقع ({prop_lat:.5f}, {prop_lon:.5f})...", icon="⏳")
        
        with st.spinner("جاري استدعاء النموذج وتحضير النتائج..."):
            # 1. Prepare Context
            analysis_context, proposed_location_tuple, competitors_list = prepare_enhanced_context_for_llm(
                neighborhood_profile, licenses_data, pois_data,
                selected_activity, prop_lat, prop_lon
            )

            # 2. Get LLM Analysis
            analysis_text = "فشل في الحصول على التحليل."
            if analysis_context:
                analysis_text = get_athar_analysis_enhanced(client, analysis_context)

            # 3. Create Output Map
            output_map_object = None
            if proposed_location_tuple:
                output_map_object = create_integrated_output_map(
                     neighborhood_profile, pois_data, competitors_list,
                     proposed_location_tuple, selected_activity
                )

        # 4. Display Results
        st.subheader("التوصية والتحليل:")
        st.markdown(analysis_text) # Display LLM analysis as markdown

        st.divider()

        st.subheader("الخريطة التفاعلية للنتائج:")
        if output_map_object:
            # Use st_folium again to render the output map
            st_folium(output_map_object, 
                      key="output_map_display", # Different key for output map
                      height=500, 
                      width=700,
                      returned_objects=[]) # No need to return anything from output map
        else:
            st.error("لم يتم إنشاء الخريطة النهائية.", icon="🗺️")

    else:
        # Handle missing prerequisites before analysis
        if not selected_activity: st.error("الرجاء اختيار نشاط تجاري من الشريط الجانبي أولاً.", icon="👈")
        if prop_lat is None or prop_lon is None: st.error("الرجاء تحديد الموقع المقترح على الخريطة في الشريط الجانبي أولاً.", icon="👈")
        if not client: st.error("خطأ في تهيئة OpenAI. يرجى التحقق من المفتاح السري.", icon="🔑")
        if not data_loaded_successfully: st.error("فشل تحميل البيانات الأساسية. لا يمكن إجراء التحليل.", icon="💾")

else:
    st.info("اختر نشاطًا وحدد موقعًا من الشريط الجانبي ثم اضغط على زر التحليل.")

st.sidebar.markdown("---")
st.sidebar.markdown("*بيانات وهمية لأغراض العرض.*")