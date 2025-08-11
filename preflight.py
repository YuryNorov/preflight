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

def accelerate_stop_distance(temperature_f, pressure_altitude_ft, gross_weight_lbs, headwind_mph=0):
    """
    Complete accelerate-stop distance formula for PA-23-250 Aztec C
    Tolerant to negative density altitudes (below sea level conditions)

    Parameters:
    temperature_f: Temperature in Fahrenheit (-60 to 140°F)
    pressure_altitude_ft: Pressure altitude in feet (-2,000 to 15,000 ft)
    gross_weight_lbs: Gross weight in pounds (3,000 to 5,600 lbs)
    headwind_mph: Headwind component in mph (0 to 25 mph)

    Returns:
    Accelerate-stop distance in feet
    """

    # Standard reference conditions
    TEMP_STD = 59.0      # °F (ISA standard at sea level)
    ALT_STD = 0.0        # ft
    WEIGHT_STD = 4400.0  # lbs (mid-range weight)
    BASE_DISTANCE = 1800.0  # ft

    # Input validation and bounds checking
    temperature_f = max(-60, min(140, temperature_f))
    pressure_altitude_ft = max(-2000, min(15000, pressure_altitude_ft))
    gross_weight_lbs = max(3000, min(5600, gross_weight_lbs))
    headwind_mph = max(0, min(25, headwind_mph))

    # Temperature factor (air density effect)
    # Based on ideal gas law: ρ ∝ 1/T
    temp_rankine = temperature_f + 459.67
    temp_std_rankine = TEMP_STD + 459.67
    temp_factor = temp_rankine / temp_std_rankine

    # Altitude factor (pressure altitude effect) - Handle negative altitudes
    # Standard atmosphere model modified for negative altitudes
    if pressure_altitude_ft >= -2000:  # Extended range for below sea level
        # Use modified standard atmosphere that handles negative altitudes
        # For negative altitudes, air density increases above sea level standard
        altitude_ratio = 1.0 - 6.8756e-6 * pressure_altitude_ft

        # Ensure altitude_ratio stays positive and reasonable
        altitude_ratio = max(0.1, min(1.5, altitude_ratio))

        # Calculate density factor (inverse relationship)
        # Higher density (negative DA) = better performance (lower factor)
        altitude_factor = altitude_ratio ** (-4.2561)

        # Additional correction for very negative altitudes
        if pressure_altitude_ft < 0:
            # Enhanced performance below sea level
            negative_alt_bonus = abs(pressure_altitude_ft) / 10000.0
            altitude_factor *= (1.0 - negative_alt_bonus * 0.1)
            altitude_factor = max(0.5, altitude_factor)  # Prevent over-optimization

    elif pressure_altitude_ft <= 36089:  # Normal positive altitude range
        altitude_ratio = 1.0 - 6.8756e-6 * pressure_altitude_ft
        altitude_factor = altitude_ratio ** (-4.2561)
    else:
        # Above tropopause (simplified)
        altitude_factor = 4.0  # Severe performance degradation

    # Weight factor (kinetic energy effect)
    # Distance ∝ weight (more energy to dissipate)
    weight_factor = gross_weight_lbs / WEIGHT_STD

    # Headwind factor (relative airspeed effect)
    # Each mph reduces ground roll distance
    headwind_factor = 1.0 - (headwind_mph * 0.025)
    headwind_factor = max(0.4, min(1.0, headwind_factor))  # Reasonable limits

    # Non-linear corrections observed from chart
    # Modified to handle extreme conditions gracefully
    temp_deviation = temperature_f - TEMP_STD
    temp_correction = 1.0 + (temp_deviation / 80.0) ** 2 * 0.08

    # Altitude correction - handle negative altitudes
    if pressure_altitude_ft >= 0:
        altitude_correction = 1.0 + (pressure_altitude_ft / 8000.0) ** 1.2 * 0.12
    else:
        # Negative altitude provides performance benefit
        altitude_correction = 1.0 - (abs(pressure_altitude_ft) / 8000.0) ** 0.8 * 0.08
        altitude_correction = max(0.7, altitude_correction)

    weight_deviation = gross_weight_lbs - WEIGHT_STD
    weight_correction = 1.0 + (weight_deviation / 1000.0) ** 2 * 0.05

    # Calculate distance with all factors
    distance = (BASE_DISTANCE *
                temp_factor *
                altitude_factor *
                weight_factor *
                headwind_factor *
                temp_correction *
                altitude_correction *
                weight_correction)

    # Apply reasonable bounds with extended range for extreme conditions
    distance = max(600, min(8000, distance))

    return round(distance)

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

def takeoff_50_nowind(w, da) :
    dist = (1.317919e-03
            - 9.248807e-01 * da
            + 2.010712e+00 * w
            - 1.382750e-04 * da**2
            + 6.107982e-04 * da*w
            - 7.011730e-04 * w**2
            + 4.315973e-09 * da**3
            + 2.147113e-08 * da**2*w
            - 8.016888e-08 * da*w**2
            + 7.058259e-08 * w**3)
    return int(dist)

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

def headwind_takeoff(w, t) :
    d =   (- 1.345024e+02 + 1.212677e+00 * t - 1.580498e+00 * w
          - 1.001003e-04 * t**2
          + 1.770352e-02 * w**2
          - 2.179669e-02 * t*w
          + 1.417450e-08 * t**3
          + 7.076594e-07 * t**2*w
          + 3.379797e-04 * t*w**2
          - 1.500000e-02 * w**3)
    return int(d)

def headwind_land(w, l):
    dist = (-7.384758e+02
            + 2.398023e+00 * l
            - 4.749474e-01 * w
            - 8.651134e-04 * l**2
            - 1.528699e-02 * l*w
            - 3.277356e-02 * w**2
            + 1.751042e-07 * l**3
            + 2.111363e-06 * l**2*w
            + 3.409930e-06 * l*w**2
            + 2.424242e-03 * w**3)
    return int(dist)

def rev_name(name):
    name_dir = (int(name[:2]) + 18) % 36
    if name_dir == 0:
        name_dir = 36

    if len(name) <= 2:
        return str(name_dir)

    if name[2] == 'L':
        name = str(name_dir) + 'R'
    else:
        name = str(name_dir) + 'L'

    return name


def __print_performance(d, weight, to_nw, land_nw, rw, rev):
    length = int(rw.length_ft)
    name = rw.le_ident
    if rev == False:
        le = float(rw.he_heading_degT)
    else:
        le = (float(rw.he_heading_degT) + 180) % 360
        name = rev_name(name)

    headwind  = int(d['wind_speed_kt'] * math.cos(math.radians(d['wind_direction'] - le)))
    crosswind = int(d['wind_speed_kt'] * math.sin(math.radians(d['wind_direction'] - le)))
    if crosswind > 0:
        crosswind = str(crosswind) + 'R'
    elif crosswind < 0:
        crosswind = str(-crosswind) + 'L'

    to = headwind_takeoff(headwind * 1.15, to_nw) 
    land = headwind_land(headwind * 1.15, land_nw)

    headwind_gust  = int(d['wind_gust_kt'] * math.cos(math.radians(d['wind_direction'] - le)))
    crosswind_gust = int(d['wind_gust_kt'] * math.sin(math.radians(d['wind_direction'] - le)))
    to_gust = headwind_takeoff(headwind_gust * 1.15, to_nw) 
    land_gust = headwind_land(headwind_gust * 1.15, land_nw) 

    tf = (d['temperature_c'] * 9/5 ) + 32
    start_stop_calm = accelerate_stop_distance(tf, d['pressure_inhg'], weight)
    start_stop      = accelerate_stop_distance(tf, d['pressure_inhg'], weight, headwind_gust * 1.15)
    start_stop_gust = accelerate_stop_distance(tf, d['pressure_inhg'], weight, headwind_gust * 1.15)

    print(f"RW  HW  CW Length:")
    print(f"{name:3} {headwind:3} {crosswind:3} {length:>5}")
    print(f"{'Takeoff':15} {to_nw:6,.0f} {to:6,.0f} {to_gust:6,.0f}")
    print(f"{'Landing':15} {land_nw:6,.0f} {land:6,.0f} {land_gust:6,.0f}")
    print(f"{'Start-stop':15} {start_stop_calm:6,.0f} {start_stop:6,.0f} {start_stop_gust:6,.0f}")
    print('─' * 36)

def print_performance(d, weight, to_nw, land_nw, rw):
    if rw.closed == 1:
        return

    if rw.le_ident[-1] == 'W':
        return
    __print_performance(d, weight, to_nw, land_nw, rw, False)
    __print_performance(d, weight, to_nw, land_nw, rw, True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("ident", help="ICAO airport ID")
    # TODO: add a logic to enforce runway
    parser.add_argument("--runway", help="Runway in use", type = int, default = 0) 

    parser.add_argument("-f", help="Front raw passangers weight in lbs", default = 200, type = int)
    parser.add_argument("-m", help="Middle raw passangers weight in lbs", default = 0, type = int)
    parser.add_argument("-r", help="Middle raw passangers weight in lbs", default = 0, type = int)
    parser.add_argument("-g", help="Gasoline, main in gallons", default = 140, type = int)
    parser.add_argument("-G", help="Gasoline, aux  in gallons", default = 36, type = int)
    parser.add_argument("-b", help="Rear baggage in lbs", default = 10, type = int)
    parser.add_argument("-B", help="Front baggage in lbs", default = 10, type = int)

    args = parser.parse_args()
    ident = args.ident.upper()
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
    if airport.size == 0:
        airport = airports[airports.ident == ident[1:]]

    elevation = float(airport.elevation_ft.values[0])

    da = density_altitude(elevation, d['pressure_inhg'], d['temperature_c'])
    pa = pressure_altitude(elevation, d['pressure_inhg'])
    to_nw = takeoff_50_nowind(weight, da)
    land_nw = landing_50_nowind(weight, da)

    headwind  = int(d['wind_speed_kt'] * math.cos(math.radians(d['wind_direction'] - runway*10)))
    crosswind = int(d['wind_speed_kt'] * math.sin(math.radians(d['wind_direction'] - runway*10)))

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

    runways = pd.read_csv("runways.csv")
    rws = runways[runways.airport_ident == ident]
    if rws.size == 0:
        rws = runways[runways.airport_ident == ident[1:]]

    print(f'\n{'Performance'}: {'Calm':>9} {'Wind':>6} {'Gusts':>6}')
    print('─' * 36)
    for index, rw in rws.iterrows():
        print_performance(d, weight, to_nw, land_nw, rw)
