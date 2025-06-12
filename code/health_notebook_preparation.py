import pandas as pd 
import numpy as np 
import os
import seaborn as sns 
import matplotlib.pyplot as plt
from datetime import date

# read in main data
df1= pd.read_csv("/Users/RyanMiner/Library/Mobile Documents/com~apple~CloudDocs/Github/apple-health-analysis/data/source csvs/HealthAutoExport-2025-03-06-2025-06-04.csv", index_col=False)

# read in sleep data 
df2 = pd.read_csv("/Users/RyanMiner/Library/Mobile Documents/com~apple~CloudDocs/Github/apple-health-analysis/data/source csvs/Sleep Analysis 2.csv")

# Drop unneeded main columns
df1 = df1.drop([
    'Caffeine (mg)', 'Basal Body Temperature (ºF)', 'Blood Alcohol Content (%)',
    'Apple Sleeping Wrist Temperature (ºF)', 'Alcohol Consumption (count)',
    'Blood Glucose (mg/dL)', 'Blood Oxygen Saturation (%)',
    'Blood Pressure [Systolic] (mmHg)', 'Blood Pressure [Diastolic] (mmHg)',
    'Body Fat Percentage (%)', 'Body Mass Index (count)', 'Calcium (mg)',
    'Carbohydrates (g)', 'Chloride (mg)', 'Cholesterol (mg)', 'Chromium (mcg)',
    'Copper (mg)', 'Cycling Cadence (count/min)', 'Cycling Distance (mi)',
    'Cycling Functional Threshold Power (watts)', 'Cycling Power (watts)',
    'Cycling Speed (mi/hr)', 'Dietary Biotin (mcg)', 'Dietary Energy (kcal)',
    'Dietary Sugar (g)', 'Distance Downhill Snow Sports (mi)',
    'Electrodermal Activity (S)', 'Environmental Audio Exposure (dBASPL)',
    'Fiber (g)', 'Flights Climbed (count)', 'Folate (mcg)',
    'Forced Expiratory Volume 1 (L)', 'Forced Vital Capacity (L)',
    'Handwashing (s)', 'Headphone Audio Exposure (dBASPL)', 'Height (cm)',
    'Inhaler Usage (count)', 'Insulin Delivery (IU)', 'Iodine (mcg)',
    'Iron (mg)', 'Lean Body Mass (lb)', 'Magnesium (mg)', 'Manganese (mg)',
    'Mindful Minutes (min)', 'Molybdenum (mcg)', 'Monounsaturated Fat (g)',
    'Niacin (mg)', 'Number of Times Fallen (falls)', 'Pantothenic Acid (mg)',
    'Peak Expiratory Flow Rate (L/min)', 'Peripheral Perfusion Index (%)',
    'Physical Effort (MET)', 'Polyunsaturated Fat (g)', 'Potassium (mg)',
    'Protein (g)', 'Push Count (count)', 'Resting Energy (kcal)',
    'Resting Heart Rate (bpm)', 'Riboflavin (mg)', 'Running Ground Contact Time (ms)',
    'Running Power (watts)', 'Running Speed (mi/hr)', 'Running Stride Length (m)',
    'Running Vertical Oscillation (cm)', 'Saturated Fat (g)', 'Selenium (mcg)',
    'Sexual Activity [Unspecified] (times)', 'Sexual Activity [Protection Used] (times)',
    'Sexual Activity [Protection Not Used] (times)', 'Six-Minute Walking Test Distance (m)',
    'Sodium (mg)', 'Stair Speed: Down (ft/s)', 'Stair Speed: Up (ft/s)',
    'Step Count (steps)', 'Swimming Distance (yd)', 'Swimming Stroke Count (count)',
    'Thiamin (mg)', 'Time in Daylight (min)', 'Toothbrushing (s)', 'Total Fat (g)',
    'Underwater Depth (ft)', 'Underwater Temperature (ºF)', 'Vitamin A (mcg)',
    'Vitamin B12 (mcg)', 'Vitamin B6 (mg)', 'Vitamin C (mg)', 'Vitamin D (mcg)',
    'Vitamin E (mg)', 'Vitamin K (mcg)', 'Waist Circumference (in)',
    'Walking + Running Distance (mi)', 'Walking Asymmetry Percentage (%)',
    'Walking Double Support Percentage (%)', 'Walking Heart Rate Average (bpm)',
    'Walking Speed (mi/hr)', 'Walking Step Length (in)', 'Water (fl. oz.)',
    'Weight/Body Mass (lb)', 'Wheelchair Distance (mi)'
], axis=1)
# print(df1.columns)


# rename date column 
df2['Date'] = df2['Date/Time']
# drop unneeded sleep columns
df2 = df2.drop([
    'Date/Time','Start','End','Asleep (Unspecified) (hr)', 
    'In Bed (hr)', 'Core (hr)', 'Deep (hr)','REM (hr)', 
    'Awake (hr)', 'Sources'
], axis = 1)
# print(df2.columns)


# merge sleep analysis and health data
df_merged = df1.merge(df2, on='Date', how='left')
#print(df_merged.columns)

# New dataframe with no date 
df_merged_nodate = df_merged.drop(columns= ['Date'])

# find means of all columns to identify empty
# create series for means of values
mean_series = df_merged_nodate.mean()

# find rows with NaN
rows_with_nan = []
for index, value in mean_series.items():
    if pd.isna(value):
        rows_with_nan.append(index)

# drop rows with NaN from merged
for row in rows_with_nan:
    df_merged = df_merged.drop(columns=[row])
# set index 
df_merged.set_index('Date',inplace=True)

#print(df_merged.columns)

# exploratory data analysis 

# additional cleaning 
# remove rows where date is between 2025-03-06 and 2025-04-02 
dates_to_drop = df_merged.loc['2025-03-06':'2025-04-02'].index
df_merged = df_merged.drop(dates_to_drop)

# change date to not show time 
df_merged.index = pd.to_datetime(df_merged.index).normalize()

# missingness 

#print(df_merged.isna().sum())
# map out missingness 

# plt.figure(figsize=(12,6))
# sns.heatmap(df_merged, cbar=False, cmap='binary',vmin=0,vmax=1)
# plt.xticks(rotation=45,ha='right')
# plt.tight_layout()
# plt.title("Missing Data Heatmap")
# plt.xlabel("Columns")
# plt.show()

# run predictive model for: 
    # Respiratory Rate (count/min),
    # Sleep Analysis [Total] (hr),
    # Sleep Analysis [Core] (hr),
    # Sleep Analysis [Deep] (hr),
    # Sleep Analysis [REM] (hr), 
    # VO2 Max (ml/(kg·min))


target_cols = [
    'Respiratory Rate (count/min)',
    'Sleep Analysis [Total] (hr)',
    'Sleep Analysis [Core] (hr)',
    'Sleep Analysis [Deep] (hr)',
    'Sleep Analysis [REM] (hr)',
    'Sleep Analysis [Awake] (hr)',
    'VO2 Max (ml/(kg·min))'
]

############################ Modeling ##################################################
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

for x in target_cols:
    # Drop rows where the target is missing (training only)
    train_df = df_merged[df_merged[x].notna()]
    X_train = train_df.drop(columns=[x])
    y_train = train_df[x]

    # Build a pipeline to impute and model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', LinearRegression())
    ])
    
    # Fit model
    pipeline.fit(X_train, y_train)

    # Predict where target is missing
    missing_df = df_merged[df_merged[x].isna()]
    X_missing = missing_df.drop(columns=[x])
    y_pred = pipeline.predict(X_missing)

    # Fill in missing values
    df_merged.loc[df_merged[x].isna(), x] = y_pred

# show missingness heatmap after modeling 
plt.figure(figsize=(12,6))
sns.heatmap(df_merged, cbar=False, cmap='binary',vmin=0,vmax=1)
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.title("Missing Data Heatmap")
plt.xlabel("Columns")
#plt.show()

# normalize data

# rule of thumb for each health metric: 
            # if higher is better

            # def normalize(metric_value):
            #     return min(max((metric_value - healthy_min) / (healthy_max - healthy_min), 0), 1)

###########     ###########      ###########        ###########      #############
            # if lower is better, invert the scale:

            # def normalize_inverse(metric_value):
            #     return min(max((healthy_max - metric_value) / (healthy_max - healthy_min), 0), 1)

''' 
Other notes: 
Max Heart Rate is hard to normalize; 
    - Penalizing the maximums penalizes exercise but captures heart irregularity
    - A higher average heart rate is likely due to exercise, so a high or low average can't be penalized 
    - Decided to only take min heart rate, hope to capture exercise data through active k/cals and active minutes  

'''



################################ Exercise Time ####################################

def normalize_exercise_time(extime):
    '''
    The 25th percentile seems to capture most of the drop off. . . 
    But an upper bound of 75 would over-reward moderately good days. 
    Decided to use 25th as lower bound and 90th as upper
    -- 25th percentile is low bound
    -- 90th percentile is upper bound
    '''
    healthy_min = df_merged['Apple Exercise Time (min)'].quantile(0.25) 
    healthy_max = df_merged['Apple Exercise Time (min)'].quantile(0.90)

    normalized = (extime - healthy_min) / (healthy_max - healthy_min) 
    return min(max(normalized, 0), 1)
df_merged['N - Apple Exercise Time (min)'] = df_merged['Apple Exercise Time (min)'].apply(normalize_exercise_time)

############################################# More = Better ######################################################

more_better = [
        'Active Energy (kcal)',
        'VO2 Max (ml/(kg·min))', 
        'Heart Rate Variability (ms)',
        'Apple Stand Time (min)',
        'Sleep Analysis [Total] (hr)',
        'Sleep Analysis [Core] (hr)',
        'Sleep Analysis [Deep] (hr)',
        'Sleep Analysis [REM] (hr)'
             ]


for x in more_better:
    def normalize(value):

        healthy_min = df_merged[x].quantile(0.05)
        healthy_max = df_merged[x].quantile(0.95)

        normalized = (value - healthy_min) / (healthy_max - healthy_min) 
 
        return min(max(normalized, 0), 1)
    df_merged['N - ' + x] = df_merged[x].apply(normalize)

######################################## Less = Better ##########################################

less_better = [
    'Heart Rate [Min] (bpm)',
    'Respiratory Rate (count/min)'
]

for x in less_better:
    def normalize(value):

        healthy_min = df_merged[x].quantile(0.05)
        healthy_max = df_merged[x].quantile(0.95)

        normalized = (healthy_max - value) / (healthy_max - healthy_min) 
 
        return min(max(normalized, 0), 1)
    df_merged['N - ' + x] = df_merged[x].apply(normalize)

### Notes from Jun 9: 

'''
- Determine which spreads to increase 
- Sort out negative (less better) variables
- Rewrite functions to accommodate 
'''

############################## APPLY WEIGHTS ##############################

'''
Asked for ChatGPT's help in determining weights for each health metric based on academic research
'''

# Determined Weights: 
weights = {
    'VO2 Max (ml/(kg·min))': 0.25,
    'Heart Rate [Min] (bpm)': 0.10,
    'Heart Rate Variability (ms)': 0.10,
    'Respiratory Rate (count/min)': 0.05,
    'Active Energy (kcal)': 0.15,
    'Apple Stand Time (min)': 0.05,
    'Sleep Analysis [Core] (hr)': 0.10,
    'Sleep Analysis [Deep] (hr)': 0.10,
    'Sleep Analysis [REM] (hr)': 0.10
}

# Prefix for normalized columns
normalized_prefix = 'N - '

# Create a health score column
df_merged['Health Score'] = 0

# Loop through each weighted metric and calculate the weighted sum
for metric, weight in weights.items():
    norm_col = normalized_prefix + metric
    df_merged['Health Score'] += df_merged[norm_col] * weight

# Save finished dataset to csv for analysis
# Today's date 
today_str = date.today().isoformat()

csvname = 'cleaned_health_data_' + today_str+ '.csv'
df_merged.to_csv(csvname)

print("Outputted Data to csv file: " + csvname)