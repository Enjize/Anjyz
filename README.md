Anjyz – Smart Business Opportunity Analyzer 
Anjyz is an AI-powered prototype web application built with Streamlit to help entrepreneurs analyze the viability of opening a specific business type at a selected location within the Al Yasmin district in Riyadh. The platform leverages real-time spatial and demographic data to support informed, data-driven investment decisions.

This is a demo using fictional data. Do not use it for real-world investment decisions.

What Is Anjyz?
Anjyz is a smart investment assistant that simulates how technology can guide business owners in making better location-based decisions. It uses:

Interactive maps

Demographic and economic data

Existing business license information

Local points of interest (POIs)

GPT-4o to generate AI-backed recommendations

Key Features
Map-Based Location Selection
Click anywhere on the interactive map to choose your proposed business location.

Business Type Dropdown
Choose from a dynamic list of business types derived from mock commercial license data.

AI Investment Recommendation
Using OpenAI’s GPT-4o, Anjyz generates investment insights based on:

Population size and income levels

Competition density and proximity

Nearby amenities such as schools, clinics, and shopping centers

Competitor Mapping
Visualize businesses with the same activity type near your chosen location.

POI Analysis
See public services and community infrastructure that influence business success.

Project Structure
File	Purpose
streamlit_app.py	Main application logic and UI built with Streamlit
yasmen.json	Simulated neighborhood profile including population, roads, and infrastructure
fake_licenses_SA-RIY-YAS_openai.json	Mock database of commercial licenses in Al Yasmin
fake_pois_SA-RIY-YAS_openai.json	Fictional list of POIs like schools, clinics, and markets
requirements.txt	Required Python packages for running the app

Setup Instructions
1. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
2. Set Your OpenAI API Key
If running locally, create a .env file in the project root with the following:

env
Copy
Edit
OPENAI_API_KEY=your-api-key-here
For cloud deployment (e.g., Streamlit Cloud), use st.secrets.

3. Run the Application
bash
Copy
Edit
streamlit run streamlit_app.py
💡 How It Works
Select a business type (e.g., restaurant, clinic, cafe).

Click on the map to choose a location in Al Yasmin.

Click "Analyze Opportunity".

Anjyz will:

Load nearby competitors and POIs

Analyze demographics and income levels

Use GPT-4o to generate a professional recommendation

View results on an interactive map with markers for competitors and nearby POIs.

🧪 Example Use Case
Scenario:
You want to open a café targeting young adults. You select "Cafés" and click a location on Anas Ibn Malik Road.

Anjyz Response:

"This location is promising due to the high concentration of residents aged 18–35, above-average income levels, and few nearby cafés. Nearby services such as schools and banks increase foot traffic potential."

🧰 Technologies Used
Python 3.11

Streamlit (web UI)

Folium + streamlit-folium (interactive maps)

OpenAI GPT-4o (LLM-based analysis)

Geopy (distance calculations)

dotenv (local API key handling)

⚠️ Disclaimer
This project uses synthetic data for demonstration purposes only.
Do not rely on the results for real financial or business decisions.

🧾 License
This project is for academic, educational, and demonstrative use only.
