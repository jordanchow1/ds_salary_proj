 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:34:40 2020

@author: Jordan Chow
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

# Parsing 

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2

df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

## State
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[-1])
df['job_state'] = df['job_state'].apply(lambda x: "NY" if x == 'New York State' else x)
df['job_state'] = df['job_state'].apply(lambda x: "NJ" if x == 'New Jersey' else x)
df['job_state'] = df['job_state'].apply(lambda x: x.split(' ')[1] if ' ' in x else x)
df['job_state'] = df['job_state'].apply(lambda x: "US" if x == "States" else x)
df['job_state'] = df['job_state'].apply(lambda x: "MD" if x == "Maryland" else x)
df['job_state'] = df['job_state'].apply(lambda x: "VA" if x == "Virginia" else x)
df.job_state.value_counts()

## Age
df['age'] = df['Founded'].apply(lambda x: x if x<1 else 2020-x)

# Feature Engineering for key skills
df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['R'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


df.to_csv('salary_data_cleaned.csv', index = False)