#!/usr/bin/env python
# coding: utf-8

import pandas as pd
#import numpy as np
import requests
import json
import plotly.express as px
from pandas import json_normalize
import datetime
import time
from io import StringIO
import streamlit as st
import plotly.graph_objects as go
pd.set_option('max_rows',None)
pd.set_option('display.max_columns', None)
st.set_page_config(layout="wide")

#Username + Password inputs
user = st.secrets["username"]
pw = st.secrets["password"]

@st.cache(allow_output_mutation=True,ttl=300)
def load_data():
    #Authenticate the user
    s = requests.Session()
    payload = {'username_or_email': user, 'password':pw}
    s.post('https://api.onepeloton.com/auth/login', json=payload)

    #Get User ID to pass into other calls
    me_url = 'https://api.onepeloton.com/api/me'
    response = s.get(me_url)
    apidata = s.get(me_url).json()

    #Get my user id
    df_my_id = json_normalize(apidata, 'id', ['id'])
    df_my_id_clean = df_my_id.iloc[0]
    my_id = (df_my_id_clean.drop([0])).values.tolist()

    #Get data from csv
    url = 'https://api.onepeloton.com/api/user/{}/workout_history_csv?timezone=America/Chicago'.format(*my_id)
    response = s.get(url)
    df = pd.read_csv(StringIO(response.text))

    return df

df = load_data()

# Isolate to just cycling rides / workouts with outputs
df_rides = df[df["Fitness Discipline"] == "Cycling"]
df_rides = df_rides.dropna(subset=['Total Output']) #dropping rides with null output values

# Clean up some of the columns
df_rides['Avg. Resistance'] = df_rides["Avg. Resistance"].str.replace("%","")
df_rides['Avg. Resistance'] = df_rides["Avg. Resistance"].astype(float)
df_rides['Workout Timestamp']= df['Workout Timestamp'].str.replace(r"\(.*\)","")
df_rides['Class Timestamp']= df['Class Timestamp'].str.replace(r"\(.*\)","")
df_rides['Workout Timestamp'] = df_rides['Workout Timestamp'].astype('datetime64')
df_rides['Class Timestamp'] = df_rides['Class Timestamp'].astype('datetime64')
#not going to convert EDT timestamps to central, only looking at timestamps by the day and I don't do super early or late workouts
df_rides = df_rides.sort_values(by='Length (minutes)')
df_rides['Length (minutes)'] = df_rides['Length (minutes)'].astype(str)
df_rides['Hour'] = df_rides['Workout Timestamp'].dt.strftime('%H')
df_rides['Hour'] = df_rides['Hour'].astype(int)
df_rides['Hour of Day'] = pd.cut(df_rides['Hour'],bins=[4,8,12,16,20,24],
                                labels=['4a-8a','8a-12p','12p-4p','4p-8p','8p-12a'])
df_rides.rename(columns={"Avg. Watts": "Avg. Output (Watts)","Total Output": "Total Output (Watts)",
                        "Avg. Resistance": "Avg. Resistance (%)","Length (minutes)":"Class Length (min)",
                        "Type":"Class Type"},inplace=True)


### Total Workouts Viz ###
total_workouts = len(df)

### Workout Breakdown Viz ###
cycle = len(df[df["Fitness Discipline"] == "Cycling"])
meditation = len(df[df["Fitness Discipline"] == "Meditation"])
strength = len(df[df["Fitness Discipline"] == "Strength"])
cardio = len(df[df["Fitness Discipline"] == "Cardio"])
stretching = len(df[df["Fitness Discipline"] == "Stretching"])
yoga = len(df[df["Fitness Discipline"] == "Yoga"])
walking = len(df[df["Fitness Discipline"] == "Walking"])

total = go.Figure()
total.add_trace(go.Indicator(
    mode = "number",
    value = total_workouts,
    domain = {'row': 0, 'column': 0},
    title = {'text': "Total Workouts"}))

total.update_layout(
    height=250,
    width =250#
)


### Class Type Distribution Viz ###

labels = ['Cycling','Strength','Meditation','Cardio','Stretching','Walking','Yoga']
values = [cycle, strength,meditation,cardio,stretching,walking,yoga]


breakdown = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+value',
                             insidetextorientation='radial'
                            )])

breakdown.update_layout(title_text='Class Type Distribution')


### Monthly Cost Viz ###

#isolate to just the bike rides
num_rides = len(df_rides)

bike_cost = 2255
monthly_cost = 39
total_rides = num_rides - 10 #subtract out the rides I did on my mom's bike
start = pd.to_datetime("2021-01-01")
today = pd.to_datetime("today")

#months of paying subscription
num_months = (today.year - start.year) * 12 + (today.month - start.month)+1

#current total cost
current_total_cost = bike_cost + (monthly_cost*num_months)

#rides done already this month
rides_this_month = 0
for row in df_rides["Workout Timestamp"]:
    if row.month == today.month:
        rides_this_month += 1

#avg cost per ride
avg_cost_per_ride = current_total_cost / (total_rides-rides_this_month)
#print('Average Cost per Ride this month:',avg_cost_per_ride)

# total cost last month
last_month = num_months-1
last_month_total_cost = bike_cost + (monthly_cost*last_month)

#convert created_at from unix epoch to datetime
df_rides["created_at"] = pd.to_datetime(df_rides["Workout Timestamp"],unit='s')

#calculate rides done already this month + last month
rides_last_month = 0
for row in df_rides["Workout Timestamp"]:
    if row.month == today.month - 1 or row.month == today.month:
        rides_last_month += 1

total_rides_last_month = total_rides - rides_last_month
last_month_avg_cost = last_month_total_cost / total_rides_last_month

cost = go.Figure()

cost.add_trace(go.Indicator(
    mode = "number+delta",
    value = avg_cost_per_ride,
    number = {'prefix': "$"},
    delta = {'reference': last_month_avg_cost, "valueformat": "$.2f"},
    domain = {'row': 0, 'column': 0},
    title = {'text': "Cost Per Ride This Month"}))

cost.update_layout(
    height=250,
    width =250
)


### Top Instructors Viz ###

#Calculate number of workouts per instuctor and workout type
instructors = df[['Workout Timestamp','Instructor Name','Fitness Discipline']]
instructors_grouped = instructors.groupby('Instructor Name').agg({'Workout Timestamp':'count'}).sort_values(by='Workout Timestamp',ascending=False).reset_index()
instructors_totals = instructors_grouped.rename(columns={'Workout Timestamp':'total'})
instructors_subtotals = instructors.groupby(['Instructor Name','Fitness Discipline']).agg({'Workout Timestamp':'count'}).sort_values(by='Workout Timestamp',ascending=False).reset_index()

instructors_pivot = pd.pivot_table(instructors_subtotals,values='Workout Timestamp',index='Instructor Name',
                                  columns=['Fitness Discipline'],aggfunc=sum,fill_value=0)
instructors_all = instructors_pivot.merge(instructors_totals,how='left',on='Instructor Name').sort_values(by='total',ascending=False)
top_instructors = instructors_all.head(10) #limit to top 10 instructors

bar = px.bar(top_instructors, x="Instructor Name",
              y=['Cardio', 'Cycling', 'Meditation', 'Strength', 'Stretching', 'Walking', 'Yoga'],
              title="Top Instructors",
             labels={'Instructor Name':'Instructor','value':'Number of Workouts','variable':'Type of Workout'})



### Donut Viz ###

# Calculate hours spent working out by workout type
df['Length (hours)'] = (df['Length (minutes)']/60).round(2)
total_hours = 'Total: '+ str(df['Length (hours)'].sum())
donut_values = df.groupby(['Fitness Discipline'],as_index=False).agg({'Length (hours)':'sum'})
donut_labels = df['Instructor Name'].unique()
donut = go.Figure(data=[go.Pie(labels=donut_values['Fitness Discipline'], values=donut_values['Length (hours)'], hole=.3)])
donut.update_traces(textinfo='value')
donut.update_layout(title_text='Total Hours Spent Working Out')


### Average Miles / Rides / Hours per week ###

# Create dataframe with weekly rides, hours, and miles
df_rides['Length (hours)'] = (df['Length (minutes)']/60).round(2)
df_rides["created_week"] = df_rides['Workout Timestamp'] - pd.to_timedelta(df_rides['Workout Timestamp'].dt.dayofweek, unit='d')
df_rides['created_week'] = df_rides['created_week'].dt.date
num_rides = df_rides.groupby('created_week')['Workout Timestamp'].count().reset_index()
miles = df_rides.groupby('created_week')['Distance (mi)'].sum()
hours_per_week = df_rides.groupby('created_week')['Length (hours)'].sum()
rid = num_rides.merge(miles,how='inner',on='created_week')
weekly_num = rid.merge(hours_per_week,how='inner',on='created_week')
weekly_num.rename(columns={"created_week":"Week","Workout Timestamp":"Number of Rides",
                      "Distance (mi)":"Distance Ridden (mi)","Length (hours)":"Hours Ridden"},inplace=True)


# Calculate averages and dump into cross-joined dataframe in order to show
# averages as a line trace on the same graph over time
avg_rides = (weekly_num["Number of Rides"].sum() / len(weekly_num.index)).round(decimals=1)
d = {'Number of Rides': [avg_rides]}
rid = pd.DataFrame(data=d)
avg_miles = (weekly_num["Distance Ridden (mi)"].sum() / len(weekly_num.index)).round(decimals=1)
d1 = {'Distance Ridden (mi)': [avg_miles]}
rid1 = pd.DataFrame(data=d1)
avg_hours = (weekly_num["Hours Ridden"].sum() / len(weekly_num.index)).round(decimals=1)
d2 = {'Hours Ridden': [avg_hours]}
rid2 = pd.DataFrame(data=d2)
averag = pd.merge(rid,rid1,left_index=True, right_index=True)
averages = pd.merge(averag,rid2,left_index=True, right_index=True)
weekly_try = weekly_num.copy()
weekly_try.rename(columns={"Number of Rides":"1","Distance Ridden (mi)": "2",
                         "Hours Ridden":"3"},inplace=True)
weekly_try['key'] = 0
averages['key'] = 0
news = averages.merge(weekly_try, on='key', how='outer')


### Display everything in Streamlit Dash ###

col1,col2 = st.columns(2)

with col1:
    st.title("Peloton Dashboard")
    st.plotly_chart(total)
    st.plotly_chart(cost)

with col2:
    st.title(" ")
    st.plotly_chart(breakdown,use_container_width=True)


st.plotly_chart(bar)

yax = st.selectbox(
             "Display on y-axis:",
             ('Total Output (Watts)', 'Avg. Output (Watts)','Avg. Resistance (%)','Avg. Cadence (RPM)',
              'Avg. Speed (mph)','Distance (mi)','Calories Burned'))

colorpts = st.selectbox("Color points according to:",
                            ('Class Length (min)','Instructor Name','Class Type','Hour of Day'))


if yax and colorpts:
        fig = px.scatter(df_rides, x='Workout Timestamp', y=yax,color=colorpts,
                         custom_data=['Title','Instructor Name'],title='Metrics over time')
        fig.update_traces(
        hovertemplate="<br>".join([
                "%{y}",
                "%{x}",
                "%{customdata[1]} " + "%{customdata[0]}"
            ])
            )
st.plotly_chart(fig)

st.plotly_chart(donut)

weeks = st.radio(
             "Display on y-axis:",
             ('Number of Rides', 'Distance Ridden (mi)','Hours Ridden'))


if weeks:
    lines = go.Figure()
    lines.add_trace(go.Bar(x=weekly_num['Week'], y=weekly_num[weeks],name=weeks))
    lines.add_trace(go.Scatter(x=news['Week'],y=news[weeks],mode="lines",name="Average"))

    lines.update_layout(title="Average Rides/Miles/Hours per Week")
st.plotly_chart(lines)
