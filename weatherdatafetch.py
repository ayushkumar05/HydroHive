import requests
import datetime

api_key = 'de7c46a20dc947c9e1679280fb9f6953'

# Set the location and time period
location = 'Patna, India'
num_days = 7

# Calculate the start and end dates for the time period
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=num_days)

# Build the API URL for historical weather data
api_url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat=25.5941&lon=85.1376&appid={api_key}&start={int(start_date.timestamp())}&end={int(end_date.timestamp())}&units=metric'

print(api_url)
# Make the API request
response = requests.get(api_url)
# Parse the JSON response
weather_data = response.json()
print(weather_data)

# Extract the daily weather data
daily_data = weather_data['daily']

# Loop through the daily data and print the temperature, humidity, and rainfall for each day
for day in daily_data:
    date = datetime.datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d')
    temp = day['temp']['day']
    humidity = day['humidity']
    if 'rain' in day:
        rain = day['rain']
    else:
        rain = 0.0
    print(f'{date}: Temperature = {temp}°C, Humidity = {humidity}%, Rainfall = {rain}mm')
