import requests
import json
import csv

# Set the API endpoints and parameters
weather_api_endpoint = "https://api.openweathermap.org/data/2.5/weather"
weather_api_key = "de7c46a20dc947c9e1679280fb9f6953"
water_level_api_endpoint = "https://waterdata.usgs.gov/nwis/uv"
terrain_data_endpoint = "https://maps.googleapis.com/maps/api/elevation/json"
terrain_api_key = "your_terrain_api_key"

# Set the location for which to collect data (Northern Bihar)
location = {
    "name": "Northern Bihar",
    "lat": 26.8467,
    "lon": 87.2718
}

# Set the date range for which to collect water level data
start_date = "2021-01-01"
end_date = "2021-12-31"

# Get the current weather data
weather_params = {
    "appid": weather_api_key,
    "lat": location["lat"],
    "lon": location["lon"]
}
response = requests.get(weather_api_endpoint, params=weather_params)
weather_data = json.loads(response.content)

# Get the historical water level data
water_level_params = {
    "site_no": "site_number",
    "startDt": start_date,
    "endDt": end_date,
    "format": "json"
}
response = requests.get(water_level_api_endpoint, params=water_level_params)
water_level_data = json.loads(response.content)

# Get the terrain data
terrain_params = {
    "locations": f"{location['lat']},{location['lon']}",
    "key": terrain_api_key
}
response = requests.get(terrain_data_endpoint, params=terrain_params)
terrain_data = json.loads(response.content)

# Write the collected data to a file
with open("flood_prediction_data.csv", mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(["location", "temperature", "rainfall", "water_level", "elevation"])
    writer.writerow([location["name"], weather_data["main"]["temp"], weather_data["rain"]["1h"], water_level_data["value"]["timeSeries"][0]["values"][0]["value"], terrain_data["results"][0]["elevation"]])
