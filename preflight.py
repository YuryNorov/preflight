#!/usr/bin/env python3

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

def to50(w, da):
    return 1019.5354 -1.908517e-01 * da +3.105556e-01 * w +1.228355e-05 * da**2 +4.500000e-05 * da * w -4.861111e-05 * w**2

def la_50(w, da):
    return 1234.1616 +3.393290e-02 * da +8.023611e-02 * w +8.000541e-07 * da**2 +3.395833e-06 * da*w -6.770833e-06 * w*2

def hw_takeoff(w, to):
    ret =  43.6873 + 9.639547e-01 * to - 1.336320e+01 * w + 5.836497e-06 * to**2 - 1.337426e-02 * to*w + 3.625000e-01 * w**2
    print(type(ret))
    print(ret)
    return ret

def hw_land(w, la):
    return 41.7542 +9.479427e-01 * la - 6.375855e+00 * w + 1.576817e-05 * la**2 - 8.293356e-03 * w*la + 2.727273e-02 * w**2

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
    #to_nw = takeoff_50_nowind(weight, da)
    to_nw = to50(weight, da)
    #land_nw = landing_50_nowind(weight, da)
    land_nw = la_50(weight, da)

    headwind  = int(d['wind_speed_kt'] * math.cos(math.radians(d['wind_direction'] - runway*10)))
    crosswind = int(d['wind_speed_kt'] * math.sin(math.radians(d['wind_direction'] - runway*10)))
    #to = headwind_takeoff(headwind * 1.15, to_nw) 
    to = hw_takeoff(headwind * 1.15, to_nw) 
    #land = headwind_land(headwind * 1.15, land_nw)
    land = hw_land(headwind * 1.15, land_nw)

    headwind_gust  = int(d['wind_gust_kt'] * math.cos(math.radians(d['wind_direction'] - runway*10)))
    crosswind_gust = int(d['wind_gust_kt'] * math.sin(math.radians(d['wind_direction'] - runway*10)))
    #to_gust = headwind_takeoff(headwind_gust * 1.15, to_nw) 
    to_gust = hw_takeoff(headwind_gust * 1.15, to_nw) 
    #land_gust = headwind_land(headwind_gust * 1.15, land_nw) 
    land_gust = hw_land(headwind_gust * 1.15, land_nw) 

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

    print(f"{'\nPerformace':12} {'Weight':>6} {'CG':>6} {'Calm':>6} {'Wind':>6} {'Gusts':>6}")
    print("-" * 46)
    print(f"{'Takeoff, ft':10} {weight:6,.0f} {cg:6,.1f} {to_nw:6,.0f} {to:6,.0f} {to_gust:6,.0f}")
    print(f"{'Landing, ft':10} {weight:6,.0f} {cg:6,.1f} {land_nw:6,.0f} {land:6,.0f} {land_gust:6,.0f}")
