import random
import numpy as np
import pandas as pd

USER_SAMPLES = 100 # number of user samples to generate
ACTIVITES_SAMPLES = 1000 

# user as is
USER_ID = np.arange(USER_SAMPLES)
AGE = np.random.randint(10, 80, USER_SAMPLES)
SEX = np.random.randint(0, 2, USER_SAMPLES) 
TENURE = np.random.randint(1, 61, USER_SAMPLES)
USER_TYPE = np.random.randint(0, 2, USER_SAMPLES) # personal, business
LAT = np.abs(0.8 * np.random.randn(USER_SAMPLES) + 44)
LON = 1 * np.abs(np.random.randn(USER_SAMPLES) + 16)
REVENUE = np.random.randint(0, 3, USER_SAMPLES) # bronze, silver, gold
CUSTOMER_TYPE = np.random.randint(0, 3, USER_SAMPLES) # mobile, fixed, convergent
HARDWARE_TYPE = np.random.randint(0, 2, USER_SAMPLES) # iOS, Android

# activities 
CUSTOMER_ID = np.random.randint(0, USER_SAMPLES, ACTIVITES_SAMPLES)
PART_OF_DAY = np.random.randint(0, 3, ACTIVITES_SAMPLES) # morning, day, night
DURATION = np.abs(np.array(300 * np.random.randn(ACTIVITES_SAMPLES) + 600, dtype=np.int32)) # seconds
USAGE_TYPE = np.random.randint(0, 4, ACTIVITES_SAMPLES) # calls, sms, data, mms
MEASURE = np.random.randint(1, 1000, ACTIVITES_SAMPLES) # number of sms, mms, calls, 
ORIGINATOR_SERVICE_TYPE = np.random.randint(0, 2, ACTIVITES_SAMPLES) # mobile, fixed
ORIGINATOR_LAT = np.abs(0.8 * np.random.randn(ACTIVITES_SAMPLES) + 44)
ORIGINATOR_LON = 1 * np.abs(np.random.randn(ACTIVITES_SAMPLES) + 16)
RECEIVER_SERVICE_TYPE = np.random.randint(0, 2, ACTIVITES_SAMPLES) # mobile, fixed
RECEIVER_LAT = np.abs(0.8 * np.random.randn(ACTIVITES_SAMPLES) + 44)
RECEIVER_LON = 1 * np.abs(np.random.randn(ACTIVITES_SAMPLES) + 16)

users_table = {
    'USER_ID' : pd.Series(USER_ID),
    'AGE' : pd.Series(AGE),
    'SEX' : pd.Series(SEX),
    'TENURE' : pd.Series(TENURE),
    'USER_TYPE' : pd.Series(USER_TYPE),
    'LAT' : pd.Series(LAT),
    'LON' : pd.Series(LON),
    'REVENUE' : pd.Series(REVENUE),
    'CUSTOMER_TYPE' : pd.Series(CUSTOMER_TYPE),
    'HARDWARE_TYPE' : pd.Series(HARDWARE_TYPE),
    }

activities_table = {
    'CUSTOMER_ID' : pd.Series(CUSTOMER_ID),
    'PART_OF_DAY' : pd.Series(PART_OF_DAY),
    'DURATION' : pd.Series(DURATION),
    'USAGE_TYPE' : pd.Series(USAGE_TYPE),
    'MEASURE' : pd.Series(MEASURE),
    'ORIGINATOR_SERVICE_TYPE' : pd.Series(ORIGINATOR_SERVICE_TYPE),
    'ORIGINATOR_LAT' : pd.Series(ORIGINATOR_LAT),
    'ORIGINATOR_LON' : pd.Series(ORIGINATOR_LON),
    'RECEIVER_SERVICE_TYPE' : pd.Series(RECEIVER_SERVICE_TYPE),
    'RECEIVER_LAT' : pd.Series(RECEIVER_LAT),
    'RECEIVER_LON' : pd.Series(RECEIVER_LON),
    }


users_frame = pd.DataFrame(users_table)
activities_frame = pd.DataFrame(activities_table)

print(activities_frame)
