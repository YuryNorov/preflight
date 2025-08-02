#!/usr/bin/env python3

from scipy import interpolate
import numpy as np

import sys
import argparse
import math
import requests
import re
from bs4 import BeautifulSoup

import pandas as pd

def weight_balance(front, middle, rear, fuel_main, fuel_tips, fbaggage, rbaggage):
    empty_weight = 3193.95
    empty_moment = 290084

    weight = ( empty_weight +
               front +
               middle +
               rear +
               fuel_main * 6  +
               fuel_tips * 6  +
               fbaggage +
               rbaggage
            )

    moment = ( empty_moment +
               89 * front +
               126 * middle +
               157 * rear +
               113 * fuel_main * 6 +
               116 * fuel_tips * 6 +
               10 * fbaggage +
               183 * rbaggage
            )

    return weight, moment, moment / weight

wind = np.array([ 0, 5, 10, 15])
to50 = np.array([ 980, 1170, 1390, 1650, 1920, 2280, 2950, 3810 ])
dist = np.array([[ 980,  870,  770,  670  ],
                 [ 1170, 1050, 940,  820  ],
                 [ 1390, 1240, 1110, 980  ],
                 [ 1650, 1490, 1340, 1190 ],
                 [ 1920, 1750, 1590, 1450 ],
                 [ 2280, 2060, 1860, 1690 ],
                 [ 2950, 2650, 2420, 2200 ],
                 [ 3810, 3480, 3190, 2950 ],
               ])

# https://ourairports.com/data/airports.csv
def load_airports_csv(filepath="airports.csv"):
    return pd.read_csv(filepath)

def get_airport_data(df, icao_code):
    row = df[df['ident'] == icao_code.upper()]
    if not row.empty:
        return row.iloc[0].to_dict()
    return None

def get_elevation(airport_data):
    if airport_data and 'elevation_ft' in airport_data:
        return airport_data['elevation_ft']
    return None

def fetch_metar(airport_code):
    url = f"https://aviationweather.gov/api/data/metar?ids={airport_code}&hours=0&order=id%2C-obs&sep=true"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    return None

def parse_metar(metar):
    result = {
        'airport': None,
        'time_utc': None,
        'wind_direction': 0,
        'wind_speed_kt': 0,
        'wind_gust_kt': 0,
        'variable_wind_dir': None,
        'temperature_c': None,
        'dewpoint_c': None,
        'pressure_inhg': None,
        'remarks': None
    }

    tokens = metar.split()

    # ICAO airport code (first token)
    if tokens:
        result['airport'] = tokens[0]

    # Observation time: e.g., 311956Z = 31st day, 19:56 Zulu
    time_match = re.search(r'\b(\d{2})(\d{2})(\d{2})Z\b', metar)
    if time_match:
        day, hour, minute = time_match.groups()
        result['time_utc'] = f"{day}T{hour}:{minute}Z"

    # Wind: e.g., 14012G18KT or VRB05KT
    wind_match = re.search(r'\b(\d{3}|VRB)(\d{2})(G\d{2})?KT\b', metar)
    if wind_match:
        direction, speed, gust = wind_match.groups()
        if direction == 'VRB':
            result['wind_direction'] = 0
            result['wind_speed_kt'] = 0
        else:
            result['wind_direction'] = int(direction)
            result['wind_speed_kt'] = int(speed)
        if gust:
            result['wind_gust_kt'] = int(gust[1:])
        else:
            result['wind_gust_kt'] = int(speed)

    # Variable wind direction: e.g., 180V240
    var_wind_match = re.search(r'\b(\d{3})V(\d{3})\b', metar)
    if var_wind_match:
        result['variable_wind_dir'] = f"{var_wind_match.group(1)}V{var_wind_match.group(2)}"

    # Temperature/dewpoint: e.g., 32/26 or M05/M07
    temp_match = re.search(r'\b(M?\d{2})/(M?\d{2})\b', metar)
    if temp_match:
        t, d = temp_match.groups()
        result['temperature_c'] = int(t.replace('M', '-'))
        result['dewpoint_c'] = int(d.replace('M', '-'))

    # Pressure: e.g., A3004 → 30.04 inHg
    pressure_match = re.search(r'\bA(\d{4})\b', metar)
    if pressure_match:
        result['pressure_inhg'] = float(pressure_match.group(1)[:2] + '.' + pressure_match.group(1)[2:])

    # Remarks section: everything after RMK
    remarks_match = re.search(r'\bRMK\s+(.*)', metar)
    if remarks_match:
        result['remarks'] = remarks_match.group(1).strip()

    return result

def pressure_altitude(altitude, altimeter):
    return int(altitude + 1000 * (29.92 - altimeter))

def density_altitude(elev, altimeter, oat):
    tstd =  15 - elev / 1000 * 2
    pa = pressure_altitude(elev, altimeter)

    return int(pa + (oat - tstd) * 120)

############################
###  Take-off Distance  ####
############################

def takeoff_50_nowind(weight, da) : 
    DA = np.arange(0, 9000, 1000)  # 9 values: 0 to 8000
    W = np.array([4000, 4400, 4800, 5200])  # 4 values
    DIST = np.array([[1380, 1440, 1220, 1290, 1390, 1490, 1600, 1700, 1820],
                     [1250, 1300, 1370, 1470, 1570, 1680, 1800, 1920, 2050],
                     [1400, 1450, 1540, 1650, 1800, 1950, 2100, 2250, 2400],
                     [1580, 1630, 1730, 1860, 2020, 2200, 2400, 2600, 2800]])

    # Interpolator
    interp_func = interpolate.RegularGridInterpolator((W, DA), DIST, bounds_error=False, fill_value=None)
    return int(interp_func(np.array([weight, da]))[0])

def landing_50_nowind(weight, da) : 
    DA = np.arange(0, 9000, 1000)  # 9 values: 0 to 8000
    W = np.array([4000, 4400, 4800, 5200])  # 4 values
    DIST = np.array([[1380, 1407, 1437, 1470, 1505, 1540, 1575, 1615, 1655],
                     [1480, 1510, 1540, 1575, 1610, 1650, 1690, 1730, 1775],
                     [1580, 1610, 1645, 1685, 1725, 1770, 1810, 1855, 1900],
                     [1680, 1715, 1750, 1790, 1835, 1880, 1925, 1975, 2025]])

    # Interpolator
    interp_func = interpolate.RegularGridInterpolator((W, DA), DIST, bounds_error=False, fill_value=None)
    return int(interp_func(np.array([weight, da]))[0])

def headwind_takeoff(wind_mph, to50_ft) : 
    DIST = np.array([[ 980,  870,  770,  670  ],  
                     [1170, 1050,  940,  820  ],  
                     [1390, 1240, 1110,  980  ],  
                     [1650, 1490, 1340, 1190 ],
                     [1920, 1750, 1590, 1450 ],
                     [2280, 2060, 1860, 1690 ],
                     [2950, 2650, 2420, 2200 ],
                     [3810, 3480, 3190, 2950 ]])
    
    # Grid axes
    wind = np.array([0, 5, 10, 15])  # x-axis
    to50 = np.array([980, 1170, 1390, 1650, 1920, 2280, 2950, 3810])  # y-axis
    
    # Interpolator with extrapolation allowed
    interp_func = interpolate.RegularGridInterpolator((to50, wind), DIST,
                                                      method="linear",
                                                      bounds_error=False, fill_value=None)
    return int(interp_func(np.array([to50_ft, wind_mph]))[0])

def headwind_land(wind_mph, land50_ft):
    DIST = np.array([
                 [ 1190, 1105, 1025,  950 ],  #  1  
                 [ 1270, 1190, 1105, 1020 ],  #  2
                 [ 1350, 1265, 1180, 1100 ],  #  3
                 [ 1430, 1345, 1255, 1170 ],  #  4
                 [ 1505, 1415, 1330, 1240 ],  #  5
                 [ 1600, 1500, 1400, 1300 ],  #  6
                 [ 1660, 1560, 1460, 1360 ],  #  7
                 [ 1770, 1660, 1550, 1450 ],  #  8
                 [ 1870, 1755, 1645, 1540 ],  #  9
                 [ 1990, 1880, 1770, 1660 ],  # 10
                 [ 2110, 1995, 1880, 1765 ]  # 11
             ])
    # Grid axes
    wind = np.array([0, 5, 10, 15])  # x-axis
    la50 = np.array([1190, 1270, 1350, 1430,
                     1505, 1600, 1660, 1770,
                     1870, 1990, 2110])
    
    # Interpolator with extrapolation allowed
    interp_func = interpolate.RegularGridInterpolator((la50, wind), DIST,
                                                      method="linear",
                                                      bounds_error=False, fill_value=None)
    return int(interp_func(np.array([land50_ft, wind_mph]))[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("ident", help="ICAO airport ID")
    parser.add_argument("runway", help="Runway in use", type = int)

    parser.add_argument("-f", help="Front raw passangers weight in lbs", default = 200, type = int)
    parser.add_argument("-m", help="Middle raw passangers weight in lbs", default = 0, type = int)
    parser.add_argument("-r", help="Middle raw passangers weight in lbs", default = 0, type = int)
    parser.add_argument("-g", help="Gasoline, main in gallons", default = 140, type = int)
    parser.add_argument("-G", help="Gasoline, aux  in gallons", default = 36, type = int)
    parser.add_argument("-b", help="Rear baggage in lbs", default = 10, type = int)
    parser.add_argument("-B", help="Front baggage in lbs", default = 10, type = int)

    args = parser.parse_args()
    ident = args.ident
    runway = int(args.runway)

    metar = fetch_metar(ident)
    if not metar:
        print("Failed to fetch METAR.")
        exit(1)

    d = parse_metar(metar)

    print("METAR: " + metar)

    weight, moment, cg = weight_balance(args.f, args.m, args.r, args.g, args.G, args.B, args.b)

    airports = pd.read_csv("airports.csv")
    airport = airports[airports.ident == ident]
    elevation = float(airport.elevation_ft.values[0])

    da = density_altitude(elevation, d['pressure_inhg'], d['temperature_c'])
    pa = pressure_altitude(elevation, d['pressure_inhg'])
    to_nw = takeoff_50_nowind(weight, da)
    land_nw = landing_50_nowind(weight, da)

    headwind  = int(d['wind_speed_kt'] * math.cos(math.radians(d['wind_direction'] - runway*10)))
    crosswind = int(d['wind_speed_kt'] * math.sin(math.radians(d['wind_direction'] - runway*10)))
    to = headwind_takeoff(headwind * 1.15, to_nw) 
    land = headwind_land(headwind * 1.15, land_nw)

    headwind_gust  = int(d['wind_gust_kt'] * math.cos(math.radians(d['wind_direction'] - runway*10)))
    crosswind_gust = int(d['wind_gust_kt'] * math.sin(math.radians(d['wind_direction'] - runway*10)))
    to_gust = headwind_takeoff(headwind_gust * 1.15, to_nw) 
    land_gust = headwind_land(headwind_gust * 1.15, land_nw) 

    print(f"{'\nWeather':21} {'Unit':10} {'Value'}")
    print(f"{'-'*20} {'-'*10} {'-'*8}")

    print(f"{f'{ident} elevation':20} {'ft':10} {elevation}")
    print(f"{'Wind direction':20} {'deg':10} {d['wind_direction']}")
    print(f"{'Wind speed':20} {'kt':10} {d['wind_speed_kt']}")
    print(f"{'Wind gust':20} {'kt':10} {d['wind_gust_kt'] or 0}")
    print(f"{'Temperature':20} {'°C':10} {d['temperature_c']}")
    print(f"{'Dew point':20} {'°C':10} {d['dewpoint_c']}")
    print(f"{'Pressure':20} {'inHg':10} {d['pressure_inhg']}")
    print(f"{'Pressure altitude':20} {'inHg':10} {pa}")
    print(f"{'Density altitude':20} {'inHg':10} {da}")
    print(f"{'Headwind':20} {'KT':10} {headwind}")

    if crosswind < 0:
        print(f"{'Crosswind':20} {'KT':10} {abs(crosswind)} LEFT")
    elif crosswind > 0:
        print(f"{'Crosswind':20} {'KT':10} {crosswind} RIGHT")
    else:
        print(f"{'Crosswind':20} {'KT':10} 0")

    print(f"{'\nAirplane':21} {'Unit':10} {'Value'}")
    print(f"{'-'*20} {'-'*10} {'-'*8}")
    print(f"{'Weight':20} {'lbs':10} {weight:8,.0f}")
    print(f"{'Moment':20} {'in*lbs':10} {int(moment):8,.0f}")
    print(f"{'Center Gravity':20} {'in':10} {cg:8,.1f}")

    print(f"{'\nPerformace':12} {'CG':>6} {'Calm':>6} {'Wind':>6} {'Gusts':>6}")
    print("-" * 40)
    print(f"{'Takeoff, ft':10} {cg:6,.1f} {to_nw:6,.0f} {to:6,.0f} {to_gust:6,.0f}")
    print(f"{'Landing, ft':10} {cg:6,.1f} {land_nw:6,.0f} {land:6,.0f} {land_gust:6,.0f}")
