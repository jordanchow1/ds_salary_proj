#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:43:23 2020

@author: jordan5560
"""

import glassdoor_scraper as gs
import pandas as pd
path = "/Users/jordan5560/Desktop/Projects/ds_salary_proj/chromedriver"

df = gs.get_jobs('data scientist', 1000, False, path, 10)

df.to_csv('glassdoor_jobs.csv', index=False)
