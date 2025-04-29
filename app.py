# -*- coding: utf-8 -*-
import os
import json
import gradio as gr
from openai import OpenAI
import folium
# في بداية الملف app.py، السطر السابع تقريبًا (معدل)
from folium.plugins import MarkerCluster # <--- قم بإزالة LatLngPopup من هنا

from geopy.distance import geodesic

from dotenv import load_dotenv
import os

load_dotenv()  # يحمل متغيرات البيئة من ملف .env
api_key = os.environ.get("OPENAI_API_KEY")


print("Gradio App: Importing libraries...")

# --- Configuration & Constants ---
NEIGHBORHOOD_ID_TO_ANALYZE = "SA-RIY-YAS"
# Approximate bounds (used for validation/context)
LAT_MIN, LAT_MAX = 24.79, 24.81
LON_MIN, LON_MAX = 46.62, 46.65

# File paths (assuming files are in the same directory as app.py)
PROFILE_FILE = 'yasmen.json'
LICENSES_FILE = 'fake_licenses_SA-RIY-YAS_openai.json'
POIS_FILE = 'fake_pois_SA-RIY-YAS_openai.json'

# --- OpenAI Client Setup ---
# IMPORTANT: Read API Key from Hugging Face Secrets
api_key = os.environ.get("OPENAI_API_KEY")
client = None
if api_key:
    try:
        client = OpenAI(api_key=api_key)
        print("Gradio App: OpenAI Client initialized successfully.")
        # Optional: Test connection
        # client.models.list() 
    except Exception as e:
        print(f"Gradio App Error: Failed to initialize OpenAI client - {e}")
else:
    print("Gradio App Warning: OPENAI_API_KEY secret not found!")
    # Handle cases where the key might not be set during build/testing if necessary

# --- Data Loading ---
def load_json_data(filename):
    """Loads data from a JSON file."""
    try:
        # Ensure the path is relative to the script location
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Gradio App: Data loaded successfully from {filename}")
        return data
    except FileNotFoundError:
        print(f"Gradio App Error: File not found {filename}")
        return None
    except json.JSONDecodeError:
        print(f"Gradio App Error: Invalid JSON format in {filename}")
        return None
    except Exception as e:
        print(f"Gradio App Error: Unexpected error loading {filename}: {e}")
        return None

print("Gradio App: Loading data files...")
neighborhood_profile = load_json_data(PROFILE_FILE)
licenses_data = load_json_data(LICENSES_FILE)
pois_data = load_json_data(POIS_FILE)
data_loaded_successfully = all([neighborhood_profile, licenses_data, pois_data])

if not data_loaded_successfully:
     print("Gradio App Error: Failed to load one or more essential data files. The app might not function correctly.")
     # Set data to empty structures to avoid crashing later functions
     neighborhood_profile = neighborhood_profile or {}
     licenses_data = licenses_data or []
     pois_data = pois_data or []

# --- Prepare Activity List for Dropdown ---
available_activities_list = []
if isinstance(licenses_data, list):
    seen_descs = set()
    for lic in licenses_data:
        desc = lic.get('activity_description_ar')
        if desc and desc not in seen_descs:
            available_activities_list.append(desc)
            seen_descs.add(desc)
available_activities_list.sort()
print(f"Gradio App: Found {len(available_activities_list)} unique activities.")

# --- Helper Functions (Context Prep, LLM Call, Map Creation) ---

def prepare_enhanced_context_for_llm(profile, licenses, pois, selected_activity, prop_lat, prop_lon):
    """Prepares a more detailed context including location analysis for the LLM."""
    # ...(Implementation from previous responses - ensure it uses the loaded data)...
    if not all([profile, licenses, pois, selected_activity, prop_lat, prop_lon]):
        print("Context Prep Error: Missing input data.")
        return None, None, [] # Return structure expected by main function

    neighborhood_name = profile.get('neighborhood_name_ar', 'غير متوفر')
    population = profile.get('demographics', {}).get('total_population', 'غير متوفر')
    age_dist = profile.get('demographics', {}).get('age_distribution', {})
    youth_ratio_18_35 = age_dist.get('18-35', 0.0) 
    income = profile.get('socio_economic', {}).get('estimated_avg_monthly_income_sar', 'غير متوفر')
    proposed_location = (prop_lat, prop_lon)

    competitors = []
    for lic in licenses:
        if lic.get('location', {}).get('neighborhood_id') == NEIGHBORHOOD_ID_TO_ANALYZE:
            if lic.get('activity_description_ar') == selected_activity:
                 loc = lic.get('location', {})
                 comp_lat = loc.get('latitude')
                 comp_lon = loc.get('longitude')
                 if comp_lat and comp_lon:
                     try:
                        distance = geodesic(proposed_location, (comp_lat, comp_lon)).km
                     except Exception as e:
                         distance = None 
                     competitors.append({
                         "name": lic.get('business_name_ar', 'غير معروف'),
                         "lat": comp_lat, "lon": comp_lon,
                         "distance_km": round(distance, 2) if distance is not None else None })

    competitor_count = len(competitors)
    competitors.sort(key=lambda x: x.get('distance_km') if x.get('distance_km') is not None else float('inf'))    
    nearest_competitor_info = "لا يوجد منافسون بنفس النشاط."
    if competitors and competitors[0].get('distance_km') is not None:
         nearest_competitor_info = f"أقرب منافس يبعد حوالي {competitors[0]['distance_km']:.2f} كم."

    poi_summary_by_category = {}
    important_poi_categories = ['Education', 'Healthcare', 'Shopping', 'Recreation', 'Religious', 'Services'] 
    nearby_important_pois = []
    MAX_NEARBY_DISTANCE_KM = 1.0 

    if pois:
        for poi in pois:
            category = poi.get('category', 'Unknown')
            poi_summary_by_category[category] = poi_summary_by_category.get(category, 0) + 1
            if category in important_poi_categories:
                 loc = poi.get('location', {})
                 poi_lat = loc.get('latitude')
                 poi_lon = loc.get('longitude')
                 if poi_lat and poi_lon:
                     try:
                         distance = geodesic(proposed_location, (poi_lat, poi_lon)).km
                         if distance <= MAX_NEARBY_DISTANCE_KM:
                             nearby_important_pois.append({ "name": poi.get('name_ar', 'غير معروف'), "category": category, "distance_km": round(distance, 2) })
                     except: pass 
    nearby_important_pois.sort(key=lambda x: x['distance_km'])
    poi_context_str = "ملخص نقاط الاهتمام بالحي: " + ", ".join([f"{cat}: {count}" for cat, count in poi_summary_by_category.items()])
    nearby_poi_str = f"نقاط اهتمام هامة قريبة (ضمن {MAX_NEARBY_DISTANCE_KM} كم): "
    if nearby_important_pois: nearby_poi_str += ", ".join([f"{p['name']} ({p['category']}) - {p['distance_km']:.2f} كم" for p in nearby_important_pois[:5]])
    else: nearby_poi_str += "لا يوجد ضمن المسافة المحددة."

    context = f"""
ملخص بيانات لتحليل فرصة استثمارية في حي {neighborhood_name} بمدينة الرياض:
- النشاط المطلوب: {selected_activity}
- الموقع المقترح: خط عرض {prop_lat}, خط طول {prop_lon}
- بيانات الحي:
    - عدد السكان التقديري: {population} نسمة
    - نسبة الشباب (18-35 سنة) التقديرية: {youth_ratio_18_35:.1%} ({youth_ratio_18_35*100:.1f}%)
    - متوسط الدخل الشهري التقديري: {income} ريال سعودي
- تحليل المنافسة والموقع:
    - عدد المشاريع القائمة بنفس النشاط في الحي: {competitor_count}
    - {nearest_competitor_info}
    - مواقع أول 5 منافسين (إن وجدوا): {json.dumps([{'lat': c['lat'], 'lon': c['lon'], 'dist_km': c['distance_km']} for c in competitors[:5]], ensure_ascii=False)}
- نقاط الاهتمام والسياق المحيط:
    - {poi_context_str}
    - {nearby_poi_str}
"""
    print("Gradio App: Context prepared for LLM.")
    return context, proposed_location, competitors


def get_athar_analysis_enhanced(llm_client, context_summary):
    """Sends ENHANCED context to the LLM and gets analysis."""
    # ...(Implementation from previous responses)...
    if not llm_client: return "خطأ: OpenAI Client غير مهيأ."
    if not context_summary: return "خطأ: سياق البيانات فارغ."

    analysis_prompt = f"""
أنت مساعد ذكي في مشروع "أثر" لتحليل فرص الاستثمار التجاري في السعودية.
مهمتك هي تحليل البيانات المقدمة عن حي معين، نشاط تجاري مطلوب، والموقع المقترح لهذا النشاط، ثم تقديم توصية مفصلة حول مدى مناسبة فتح هذا النشاط في الموقع المقترح ضمن هذا الحي.

البيانات المقدمة للتحليل:
{context_summary}

المطلوب:
قدم تحليلاً وتوصية واضحة، مع الأخذ في الاعتبار العوامل التالية:
1.  **حجم السوق المحتمل:** بناءً على بيانات السكان والدخل ونسبة الشباب.
2.  **مستوى المنافسة:** بناءً على عدد المنافسين الحاليين بنفس النشاط.
3.  **تحليل الموقع:** بناءً على الموقع المقترح، ومدى قربه من المنافسين (استخدم بيانات المسافة ومواقع المنافسين)، ومدى قربه من نقاط الاهتمام الهامة (مثل المدارس، الأسواق، الحدائق) التي قد تخدم النشاط أو تجذب العملاء.
4.  **ملخص نقاط الاهتمام:** لتقييم مدى حيوية المنطقة وتوفر الخدمات الأساسية.

يجب أن تكون التوصية النهائية إحدى هذه الخيارات الرئيسية مع تعليل واضح يربط بين البيانات والموقع:
1.  **فرصة واعدة في الموقع المقترح:** (منافسة قليلة/بعيدة، سوق جيد، موقع قريب من نقاط جذب/مناطق سكنية ذات صلة).
2.  **ممكن مع الأخذ بالاعتبار (اذكر الاعتبارات):** (منافسة متوسطة، قد يكون الموقع قريبًا جدًا من منافس، أو بعيدًا عن نقاط الجذب، يحتاج لدراسة أعمق للسوق المستهدف في هذا الموقع المحدد).
3.  **الموقع قد يكون غير مثالي / المنطقة مشبعة:** (منافسة عالية وقريبة، الموقع غير ملائم للنشاط، بعيد عن الخدمات).

اجعل الرد باللغة العربية وبأسلوب احترافي ومفصل بشكل معقول (ضمن 150-200 كلمة).
"""
    try:
        print("Gradio App: Sending request to LLM...")
        completion = llm_client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "أنت مساعد تحليل استثماري متخصص في السوق السعودي مع التركيز على تحليل المواقع. قدم تحليلًا وتوصية بناءً على البيانات التفصيلية المعطاة."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.6, 
            max_tokens=350 
        )
        analysis_result = completion.choices[0].message.content
        print("Gradio App: Received response from LLM.")
        return analysis_result
    except Exception as e:
        print(f"Gradio App Error: LLM API call failed - {e}")
        return f"عذرًا، حدث خطأ أثناء محاولة تحليل البيانات ({e})."


def create_integrated_output_map(profile, pois, competitors_list, proposed_loc, selected_activity):
    """Creates the final map showing proposed location, competitors, and POIs."""
    # ...(Implementation from previous responses - ensure it uses loaded data)...
    if not profile or not proposed_loc: return None        
    center_lat, center_lon = proposed_loc[0], proposed_loc[1]
    zoom_start = 15 

    output_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles='CartoDB positron')

    # Add Proposed Location Marker
    folium.Marker(
        location=proposed_loc,
        popup=f"<b>الموقع المقترح</b><br>للنشاط: {selected_activity}",
        tooltip="الموقع المقترح",
        icon=folium.Icon(color='green', icon='star', prefix='fa')
    ).add_to(output_map)

    # Add Competitors
    if competitors_list:
        comp_cluster = MarkerCluster(name=f"منافسون ({selected_activity})").add_to(output_map)
        for comp in competitors_list:
             dist_text = f"{comp['distance_km']:.2f} كم" if comp.get('distance_km') is not None else "N/A"
             popup_html = f"<b>{comp['name']}</b><br><i>منافس</i><br>المسافة: {dist_text}"
             folium.Marker( location=[comp['lat'], comp['lon']], popup=folium.Popup(popup_html, max_width=250),
                 tooltip=f"{comp['name']} (منافس)", icon=folium.Icon(color='red', icon='briefcase', prefix='fa')
             ).add_to(comp_cluster)

    # Add POIs
    if pois:
        poi_cluster = MarkerCluster(name="نقاط الاهتمام").add_to(output_map)
        for poi in pois:
           try:
                loc = poi.get('location', {})
                lat = loc.get('latitude')
                lon = loc.get('longitude')
                if lat is not None and lon is not None:
                        name = poi.get('name_ar', 'غير معروف')
                        category = poi.get('category', 'غير معروف')
                        subcategory = poi.get('subcategory', '')
                        icon_name = 'info-circle'; icon_color = 'blue'      
                        if category.lower() == 'education': icon_name = 'graduation-cap'; icon_color = 'darkblue'
                        elif category.lower() == 'healthcare': icon_name = 'hospital-o'; icon_color = 'red'
                        elif category.lower() == 'shopping': icon_name = 'shopping-cart'; icon_color = 'purple'
                        elif category.lower() == 'recreation': icon_name = 'tree'; icon_color = 'green'
                        elif category.lower() == 'religious': icon_name = 'moon-o'; icon_color = 'darkgreen'
                        elif category.lower() == 'services': icon_name = 'bank'; icon_color = 'cadetblue'
                        popup_html = f"<b>{name}</b><br>الفئة: {category} ({subcategory})"
                        folium.Marker( location=[lat, lon], popup=folium.Popup(popup_html, max_width=300),
                            tooltip=name, icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa', icon_size=(20,20))
                        ).add_to(poi_cluster) 
           except Exception as e: print(f"Map Error adding POI {poi.get('poi_id', '')}: {e}") 
        
    folium.LayerControl().add_to(output_map)
    print("Gradio App: Output map created.")
    return output_map


# --- Gradio Main Function ---
def run_athar_analysis_gradio(selected_activity, proposed_lat, proposed_lon):
    """Main function called by Gradio interface."""
    print(f"Gradio App: Received request - Activity: {selected_activity}, Lat: {proposed_lat}, Lon: {proposed_lon}")
    # Basic Input Validation
    if not selected_activity:
        return "الرجاء اختيار نشاط تجاري أولاً.", None
    if proposed_lat is None or proposed_lon is None:
        return "الرجاء إدخال خط العرض وخط الطول للموقع المقترح.", None
    # Validate coordinates are within reasonable bounds for Riyadh (optional but good)
    if not (24.0 < proposed_lat < 25.5 and 46.0 < proposed_lon < 47.5):
         return "إحداثيات الموقع المقترح تبدو غير صحيحة (خارج نطاق الرياض).", None
    # Check if data loaded
    if not data_loaded_successfully:
        return "خطأ: فشل تحميل ملفات البيانات الأساسية.", None
    # Check if API client is ready
    if not client:
        return "خطأ: لم يتم إعداد مفتاح OpenAI API بشكل صحيح.", None

    # 1. Prepare Context
    context, proposed_location_tuple, competitors_list = prepare_enhanced_context_for_llm(
        neighborhood_profile, licenses_data, pois_data, 
        selected_activity, proposed_lat, proposed_lon
    )
    
    # 2. Get LLM Analysis
    analysis_text = "فشل في الحصول على التحليل." # Default message
    if context:
        analysis_text = get_athar_analysis_enhanced(client, context)
        
    # 3. Create Output Map
    output_map = None
    if proposed_location_tuple:
        output_map = create_integrated_output_map(
             neighborhood_profile, pois_data, competitors_list, 
             proposed_location_tuple, selected_activity
        )

    print("Gradio App: Analysis and map generation complete.")
    # Gradio's gr.Plot can handle folium map objects directly
    return analysis_text, output_map 

# --- Gradio Interface Definition ---
print("Gradio App: Defining Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft(), title="Athar Investment Analyzer") as iface:
    gr.Markdown("# مشروع أثر - تحليل فرص الاستثمار التجاري")
    gr.Markdown("أداة تجريبية لتحليل مدى مناسبة فتح نشاط تجاري في **حي الياسمين بالرياض** بناءً على بيانات وهمية. أدخل النشاط والموقع المقترح للحصول على تحليل.")
    
    with gr.Row():
        with gr.Column(scale=1):
            activity_input = gr.Dropdown(
                choices=available_activities_list,
                label="1. اختر النشاط التجاري",
                info="اختر من قائمة الأنشطة المتوفرة في بيانات الحي."
            )
            map_input = Map(
                interactive=True,
                label="2. اختر موقع النشاط على الخريطة",
                value={'lat': (LAT_MIN+LAT_MAX)/2, 'lng': (LON_MIN+LON_MAX)/2},
                zoom=15
            )
            submit_button = gr.Button("🚀 تحليل الفرصة", variant="primary")

        with gr.Column(scale=2):
            analysis_output = gr.Textbox(
                label="تحليل وتوصية أثر:", 
                lines=12,
                interactive=False # Make output non-editable
            )
            map_output = gr.Plot(label="الخريطة التفاعلية للنتائج")

    # ربط المُدخلات والمخرجات
    submit_button.click(
        fn=run_athar_analysis_gradio,
        inputs=[activity_input, map_input],
        outputs=[analysis_output, map_output]
    )
    
    gr.Markdown("--- \n *ملاحظة: هذه الأداة تستخدم بيانات وهمية لأغراض العرض التوضيحي.*")

print("Gradio App: Interface defined.")

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("Gradio App: Launching interface...")
    # queue() enables handling multiple users if deployed
    # share=True creates a temporary public link (useful for testing from VS Code tunnel?)
    iface.queue().launch(debug=False) # Set debug=True for more detailed logs if needed