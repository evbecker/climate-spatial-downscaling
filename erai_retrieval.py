#!/usr/bin/env python
import os
from ecmwfapi import ECMWFDataServer


"""
API for downloading ERA-iterim data from ecmwf servers 
"""

# data request params
server = ECMWFDataServer(url="https://api.ecmwf.int/v1",key="dd2d771484943ef8d9092a9607855a00",email="evbecker@ucla.edu")
var_name = 'precip'
start_year=2009
end_year=2019

for year in range(start_year, end_year):
    print(f'YEAR IS: {year}')
    if var_name == 'precip':
        # configured to retrieve precipitation totals
        server.retrieve({
            "class": "ei",
            "dataset": "interim",
            "date": f'{year}-01-01/to/{year}-12-31',
            "expver": "1",
            "grid": "0.75/0.75",
            "levtype": "sfc",
            "param": "228.128",
            "step": "12",
            "stream": "oper",
            "time": "00:00:00/12:00:00",
            "type": "fc",
            'format'    : "netcdf",
            "target": f'erai-{year}-precip.nc',
        })
    elif var_name == 'temp':
        # configured to retrieve max 2m temperatures
        server.retrieve({
            "class": "ei",
            "dataset": "interim",
            "date": f'{year}-01-01/to/{year}-12-31',
            "expver": "1",
            "grid": "0.75/0.75",
            "levtype": "sfc",
            "param": "201.128",
            "step": "12",
            "stream": "oper",
            "time": "00:00:00/12:00:00",
            "type": "fc",
            'format'    : "netcdf",
            "target": f'erai-{year}-temp.nc',
        })