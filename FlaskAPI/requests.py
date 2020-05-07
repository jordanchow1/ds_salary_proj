#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:21:04 2020

@author: Jordan Chow
"""
import requests
from data_input import data_in

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
data = {'input': data_in}

r = requests.get(URL,headers=headers, json=data)

r.json()