from flask import Flask, request, jsonify
import sqlite3
from math import sin, cos, sqrt, atan2, radians
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

@app.route('/nearest-safepoint', methods=['POST'])
def find_nearest_safepoint():
    data = request.get_json()
    print(data)
    # Get the current live location
    current_lat = data['latitude']
    current_lon = data['longitude']

    # Connect to the SQLite3 database
    conn = sqlite3.connect('safepoints.db')
    c = conn.cursor()

    # Get all the safe points from the database
    c.execute("SELECT * FROM location")
    rows = c.fetchall()

    # Find the nearest safe point
    min_distance = float('inf')
    nearest_safepoint = None
    eh = 0
    for row in rows:
        safepoint_id, safepoint_lat, safepoint_lon, eh = rows[0]
        distance = calculate_distance(current_lat, current_lon, safepoint_lat, safepoint_lon)
        if distance < min_distance:
            min_distance = distance
            nearest_safepoint = {'id': safepoint_id, 'latitude': safepoint_lat, 'longitude': safepoint_lon}

    # Close the database connection
    conn.close()

    # Return the nearest safe point as a JSON response
    return jsonify(nearest_safepoint)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
