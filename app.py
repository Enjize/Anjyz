# -*- coding: utf-8 -*-
import gradio as gr
import folium
from folium.plugins import MarkerCluster, HeatMap, LocateControl, Draw
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from math import radians, sin, cos, sqrt, atan2
import branca.element

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Make sure it's set in the .env file.")

client = OpenAI(api_key=openai_api_key)

# --- Constants ---
YASMEN_CENTER = [24.8087, 46.6419] # Center of Al Yasmin district

# --- Helper Functions ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the earth."""
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance * 1000 # return distance in meters

def load_data(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def find_competitors(licenses_data, activity_type, target_lat, target_lon, radius_km=1.0):
    """Find competitors within a given radius."""
    competitors = []
    if not licenses_data or 'features' not in licenses_data:
        return competitors

    for feature in licenses_data['features']:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        if not properties or not geometry or geometry.get('type') != 'Point':
            continue

        # Normalize activity description for comparison (simple example)
        license_activity = properties.get('ACTIVITIE_DESCRIPTION', '').strip()
        # Consider adjusting matching logic if needed (e.g., partial match, category match)
        if activity_type.strip().lower() in license_activity.lower():
            coords = geometry.get('coordinates')
            if coords and len(coords) == 2:
                comp_lon, comp_lat = coords
                distance = haversine(target_lat, target_lon, comp_lat, comp_lon)
                if distance <= radius_km * 1000:
                    competitors.append({
                        'name': properties.get('CLIENT_NAME', 'N/A'),
                        'activity': license_activity,
                        'latitude': comp_lat,
                        'longitude': comp_lon,
                        'distance_m': round(distance)
                    })
    return sorted(competitors, key=lambda x: x['distance_m'])

def find_nearby_pois(pois_data, target_lat, target_lon, radius_km=0.5):
    """Find Points of Interest (POIs) within a given radius."""
    nearby_pois = []
    if not pois_data or 'features' not in pois_data:
        return nearby_pois

    for feature in pois_data['features']:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        if not properties or not geometry or geometry.get('type') != 'Point':
            continue

        poi_type = properties.get('fclass', 'unknown')
        poi_name = properties.get('name', 'N/A')
        coords = geometry.get('coordinates')

        if coords and len(coords) == 2:
            poi_lon, poi_lat = coords
            distance = haversine(target_lat, target_lon, poi_lat, poi_lon)
            if distance <= radius_km * 1000:
                nearby_pois.append({
                    'name': poi_name,
                    'type': poi_type,
                    'latitude': poi_lat,
                    'longitude': poi_lon,
                    'distance_m': round(distance)
                })
    return sorted(nearby_pois, key=lambda x: x['distance_m'])


def prepare_enhanced_context_for_llm(district_info, activity_type, target_lat, target_lon, competitors, nearby_pois):
    """Prepares a detailed context string for the LLM."""
    context = f"**Investment Analysis Context:**\n\n"
    context += f"*   **Target Location:** Al Yasmin District, Riyadh (Lat: {target_lat}, Lon: {target_lon})\n"
    context += f"*   **Proposed Business Activity:** {activity_type}\n\n"

    if district_info:
        context += "**District Information (Al Yasmin):**\n"
        context += f"*   Population: {district_info.get('population', 'N/A')}\n"
        context += f"*   Households: {district_info.get('households', 'N/A')}\n"
        context += f"*   Avg. Household Income: {district_info.get('avg_household_income', 'N/A')} SAR\n"
        context += f"*   Key Demographics: {district_info.get('demographics_summary', 'N/A')}\n"
        context += f"*   Economic Profile: {district_info.get('economic_profile', 'N/A')}\n\n"

    context += f"**Competitive Landscape (within 1km of target location for '{activity_type}'):**\n"
    if competitors:
        context += f"*   Number of direct competitors found: {len(competitors)}\n"
        for i, c in enumerate(competitors[:5]): # Show top 5 closest
             context += f"    - Competitor {i+1}: ~{c['distance_m']}m away\n" # Removed name for privacy if needed
        if len(competitors) > 5:
            context += "    - ... and others.\n"
    else:
        context += "*   No direct competitors found within 1km.\n"
    context += "\n"

    context += f"**Nearby Points of Interest (within 500m):**\n"
    if nearby_pois:
        poi_summary = {}
        for poi in nearby_pois:
            poi_type = poi['type']
            poi_summary[poi_type] = poi_summary.get(poi_type, 0) + 1

        context += f"*   Total POIs found: {len(nearby_pois)}\n"
        for poi_type, count in poi_summary.items():
             context += f"    - {poi_type.replace('_', ' ').title()}: {count}\n"
        # Example: Add distance to the closest school or mosque if relevant
        closest_school = min([p for p in nearby_pois if 'school' in p['type']], key=lambda x: x['distance_m'], default=None)
        if closest_school:
             context += f"*   Closest School: {closest_school['name']} (~{closest_school['distance_m']}m away)\n"
        closest_mosque = min([p for p in nearby_pois if 'mosque' in p['type']], key=lambda x: x['distance_m'], default=None)
        if closest_mosque:
             context += f"*   Closest Mosque: {closest_mosque['name']} (~{closest_mosque['distance_m']}m away)\n"

    else:
        context += "*   No significant points of interest found within 500m.\n"
    context += "\n"

    context += "**Analysis Request:**\n"
    context += f"Based on the provided district information, the proposed business activity ('{activity_type}'), the specific target location, the competitive landscape, and nearby POIs, please provide a concise analysis of the investment opportunity. Focus on:\n"
    context += f"1.  **Market Suitability:** Is '{activity_type}' suitable for this specific location within Al Yasmin, considering demographics and nearby POIs?\n"
    context += f"2.  **Competition:** How significant is the existing competition?\n"
    context += f"3.  **Potential:** What is the overall potential or risk level?\n"
    context += f"4.  **Recommendation:** Provide a brief recommendation (e.g., Proceed with caution, Good potential, High risk).\n"
    context += "Keep the analysis brief and focused on the provided data points."

    return context

def get_athar_analysis_enhanced(context):
    """Sends the context to OpenAI API for analysis."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or your preferred model
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing commercial investment opportunities based on provided location data, demographics, competitor analysis, and points of interest. Provide concise, data-driven insights."},
                {"role": "user", "content": context}
            ],
            max_tokens=400, # Adjust as needed
            temperature=0.5 # Adjust for creativity vs. factuality
        )
        # Accessing the response content correctly
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error: Could not get analysis from AI. Details: {e}"

# --- Map Creation ---
# ...existing code...
def create_integrated_output_map(target_lat, target_lon, activity_type, licenses_data, pois_data):
    """Creates a Folium map showing target, competitors, and POIs."""
    if target_lat is None or target_lon is None:
         # Return an initial map centered on Yasmin if no coords selected
         map_obj = folium.Map(location=YASMEN_CENTER, zoom_start=14, tiles='CartoDB positron')
         # Add JavaScript to update Gradio fields on click
         map_obj.add_child(folium.LatLngPopup()) # Shows coords on click

         # Custom JS to update Gradio input fields
         # Note: This relies on the specific structure/IDs Gradio generates, which might change.
         # We target the number inputs by their placeholder text as a workaround.
         js = """
            <script>
            function updateGradioCoords(lat, lon) {
                console.log("Map clicked at:", lat, lon);
                // Find the Gradio number input elements. This is fragile.
                // It assumes the Gradio interface structure won't change drastically.
                // We look for the specific number inputs used for lat/lon.
                var latInput = document.querySelector("input[type='number'][step='any'][placeholder='Latitude']"); // Adjust selector if needed
                var lonInput = document.querySelector("input[type='number'][step='any'][placeholder='Longitude']"); // Adjust selector if needed

                if (latInput && lonInput) {
                    console.log("Found input fields:", latInput, lonInput);
                    latInput.value = lat.toFixed(6);
                    lonInput.value = lon.toFixed(6);

                    // Trigger input/change events to notify Gradio of the update
                    // Need both 'input' and 'change' for Gradio to reliably pick it up.
                    var inputEvent = new Event('input', { bubbles: true });
                    var changeEvent = new Event('change', { bubbles: true });
                    latInput.dispatchEvent(inputEvent);
                    latInput.dispatchEvent(changeEvent);
                    lonInput.dispatchEvent(inputEvent);
                    lonInput.dispatchEvent(changeEvent);
                    console.log("Updated Gradio fields");

                    // Optional: Add a temporary marker (will be replaced on analysis)
                    // This requires Leaflet (L) to be available, which Folium uses.
                    // if (typeof L !== 'undefined' && typeof map !== 'undefined') {
                    //     L.marker([lat, lon]).addTo(map).bindPopup('Selected Location').openPopup();
                    // }

                } else {
                    console.error("Could not find Gradio latitude/longitude input fields.");
                }
            }

            // Add click listener to the map
            // Ensure this runs *after* the map object ('map_div_id') is created by Folium/Gradio
            // We might need to wrap this in a function called by Folium's rendering process
            // or use a MutationObserver. For simplicity, let's try adding it directly.

            // Assuming the map object created by Folium is named 'map_...'
            // We need to find the correct map object in the global scope or attach the event differently.
            // Folium map objects are often named like 'map_a1b2c3d4...'
            // A more robust way is to attach the event within the Folium map generation if possible.

            // Let's try adding the listener directly to the map div Folium creates.
            // We need the ID of the div where Folium renders the map.
            // Let's assume Gradio renders the HTML output in a div, and Folium map is inside.
            // We'll use a general approach targeting the map container.

            document.addEventListener('DOMContentLoaded', function() {
                // Find the map container (might need adjustment based on Gradio's output structure)
                var mapElement = document.querySelector('.folium-map'); // Adjust if Folium uses a different class
                if (mapElement && mapElement.__folium_map) { // Check if Folium attached the map object
                     mapElement.__folium_map.on('click', function(e) {
                         updateGradioCoords(e.latlng.lat, e.latlng.lng);
                     });
                     console.log("Map click listener attached.");
                } else {
                     // Fallback or alternative method if the above doesn't work
                     console.warn("Could not attach click listener directly to Folium map object. Trying LatLngPopup approach.");
                     // The folium.LatLngPopup() added earlier might be sufficient if it triggers updates,
                     // but we want to update specific fields.
                     // Let's rely on the custom JS within the map's HTML source.
                }
            });


            </script>
            """
         # Embed the JavaScript into the map's HTML
         # This is a bit hacky, injecting script into the rendered HTML
         map_html = map_obj.get_root().render()
         map_html_with_js = map_html.replace('</head>', f'{js}</head>') # Inject JS into head
         # Instead of returning map_obj._repr_html_(), return the modified HTML string
         # We need Gradio to render this HTML. Use gr.HTML output.
         # The function will now return HTML string instead of map object representation.
         # This requires changing the Gradio output component for the map to gr.HTML.

         # Add a simple popup instruction
         folium.Marker(
             YASMEN_CENTER,
             popup="Click on the map to select your location",
             tooltip="Click to select location",
             icon=folium.Icon(color='blue', icon='info-sign')
         ).add_to(map_obj)

         # Return the HTML representation for Gradio's HTML component
         return map_obj._repr_html_() # Still return HTML for gr.Plot or gr.HTML


    # --- Existing map generation logic when lat/lon ARE provided ---
    map_obj = folium.Map(location=[target_lat, target_lon], zoom_start=16, tiles='CartoDB positron')

    # Add marker for the target location
    folium.Marker(
        [target_lat, target_lon],
        popup=f"Proposed Location\nActivity: {activity_type}",
        tooltip="Proposed Location",
        icon=folium.Icon(color='green', icon='star')
    ).add_to(map_obj)

    # Add competitors to the map
    competitor_group = folium.FeatureGroup(name=f"Competitors ({activity_type})")
    competitors = find_competitors(licenses_data, activity_type, target_lat, target_lon)
    for comp in competitors:
        folium.Marker(
            [comp['latitude'], comp['longitude']],
            popup=f"Competitor: {comp.get('name', 'N/A')}\nActivity: {comp['activity']}\nDistance: {comp['distance_m']}m",
            tooltip=f"Competitor (~{comp['distance_m']}m)",
            icon=folium.Icon(color='red', icon='briefcase')
        ).add_to(competitor_group)
    map_obj.add_child(competitor_group)

    # Add POIs to the map
    poi_group = folium.FeatureGroup(name="Nearby POIs (500m)")
    nearby_pois = find_nearby_pois(pois_data, target_lat, target_lon)
    poi_icon_map = {
        'school': 'graduation-cap', 'hospital': 'hospital-o', 'clinic': 'medkit',
        'pharmacy': 'plus-square', 'bank': 'bank', 'atm': 'credit-card',
        'supermarket': 'shopping-cart', 'convenience': 'shopping-basket',
        'restaurant': 'cutlery', 'cafe': 'coffee', 'fast_food': 'car', # Using 'car' as placeholder
        'fuel': 'tint', 'car_wash': 'car', 'car_repair': 'wrench',
        'mosque': 'moon-o', # Using FontAwesome icons
        'police': 'shield', 'fire_station': 'fire-extinguisher',
        'hotel': 'bed', 'park': 'tree',
        # Add more mappings as needed
    }
    for poi in nearby_pois:
        icon_name = poi_icon_map.get(poi['type'], 'info-sign') # Default icon
        folium.Marker(
            [poi['latitude'], poi['longitude']],
            popup=f"POI: {poi['name']}\nType: {poi['type']}\nDistance: {poi['distance_m']}m",
            tooltip=f"{poi['type'].title()} (~{poi['distance_m']}m)",
            icon=folium.Icon(color='blue', icon=icon_name, prefix='fa' if icon_name != 'info-sign' else 'glyphicon') # Use FontAwesome prefix for specific icons
        ).add_to(poi_group)
    map_obj.add_child(poi_group)

    # Add layer control
    folium.LayerControl().add_to(map_obj)
    # Add locate control
    LocateControl().add_to(map_obj)

    # Add Draw plugin (optional, for user drawing)
    # Draw(export=True).add_to(map_obj)

    # Return HTML representation of the map
    return map_obj._repr_html_()


# --- Main Analysis Function ---
def analyze_investment(activity_type, latitude, longitude):
    """Loads data, prepares context, gets analysis, and creates map."""
    # Validate inputs
    if not activity_type:
        return "Please select a business activity.", create_integrated_output_map(None, None, None, None, None) # Return initial map
    if latitude is None or longitude is None:
        # If called without coordinates (e.g., initial load or after invalid click)
        # Return only the initial map with click instructions
        return "Please click on the map to select a location.", create_integrated_output_map(None, None, None, None, None)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
         return "Invalid latitude or longitude values.", create_integrated_output_map(YASMEN_CENTER[0], YASMEN_CENTER[1], activity_type, None, None) # Show map centered

    # Load data (consider caching if large)
    district_data = load_data('yasmen.json')
    licenses_data = load_data('fake_licenses_SA-RIY-YAS_openai.json')
    pois_data = load_data('fake_pois_SA-RIY-YAS_openai.json')

    if not district_data or not licenses_data or not pois_data:
        return "Error: Could not load necessary data files.", create_integrated_output_map(latitude, longitude, activity_type, None, None) # Show map at location

    # Find competitors and POIs
    competitors = find_competitors(licenses_data, activity_type, latitude, longitude)
    nearby_pois = find_nearby_pois(pois_data, latitude, longitude)

    # Prepare context for LLM
    context = prepare_enhanced_context_for_llm(
        district_data.get('Al Yasmin'), # Assuming district name is key
        activity_type,
        latitude,
        longitude,
        competitors,
        nearby_pois
    )

    # Get analysis from LLM
    analysis_text = get_athar_analysis_enhanced(context)

    # Create the map with results
    output_map_html = create_integrated_output_map(latitude, longitude, activity_type, licenses_data, pois_data)

    return analysis_text, output_map_html

# --- Gradio Interface ---
# Define activity choices (example list, expand as needed)
activity_choices = [
    "Supermarket", "Convenience Store", "Restaurant", "Cafe", "Pharmacy",
    "Bakery", "Laundry", "Barber Shop", "Beauty Salon", "Gym", "Bookstore",
    "Electronics Store", "Clothing Store", "Shoe Store", "Hardware Store",
    "Flower Shop", "Pet Store", "Clinic", "Car Wash", "Coffee Shop"
]

# Use gr.HTML for map output to render the Folium HTML including JavaScript
# Make lat/lon inputs non-interactive initially, they will be updated by map click
with gr.Blocks(theme=gr.themes.Soft(), title="Athar - Investment Analysis") as demo:
    gr.Markdown("# Athar - Investment Opportunity Analysis (Al Yasmin, Riyadh)")
    gr.Markdown("Select a business activity and **click on the map** to choose your proposed location.")

    with gr.Row():
        activity = gr.Dropdown(choices=activity_choices, label="Business Activity Type")
        # Make Lat/Lon inputs visible but not directly editable by user initially
        # They will display the coordinates selected from the map.
        lat_input = gr.Number(label="Latitude", placeholder="Latitude", info="Click map to set", interactive=False)
        lon_input = gr.Number(label="Longitude", placeholder="Longitude", info="Click map to set", interactive=False)

    analyze_button = gr.Button("Analyze Investment Opportunity")

    with gr.Row():
        analysis_output = gr.Markdown(label="Analysis Results")
        # Use gr.HTML to render the Folium map HTML, including our custom JS
        map_output = gr.HTML(label="Interactive Map")

    # Define the interaction: Button click triggers analysis
    analyze_button.click(
        fn=analyze_investment,
        inputs=[activity, lat_input, lon_input],
        outputs=[analysis_output, map_output]
    )

    # Load initial map on interface load
    # We need a way to trigger the map creation initially.
    # We can call analyze_investment with None values to get the initial map.
    demo.load(
        fn=lambda: ( "Please select an activity and click the map.", create_integrated_output_map(None, None, None, None, None) ),
        inputs=None,
        outputs=[analysis_output, map_output]
    )


if __name__ == "__main__":
    demo.launch()