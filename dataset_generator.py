import numpy as np
import pandas as pd

import plan_predictor

USER_SAMPLES = 100
ACTIVITIES_SAMPLES = 1000


def generate_tables(user_samples=USER_SAMPLES, activities_samples=ACTIVITIES_SAMPLES):
    # user as is
    USER_ID = np.arange(user_samples)
    AGE = np.random.randint(10, 80, user_samples)
    SEX = np.random.randint(0, 2, user_samples)
    TENURE = np.random.randint(1, 61, user_samples)
    USER_TYPE = np.random.randint(0, 2, user_samples)  # personal, business
    LAT = np.abs(0.8 * np.random.randn(user_samples) + 44)
    LON = 1 * np.abs(np.random.randn(user_samples) + 16)
    REVENUE = np.random.randint(0, 3, user_samples)  # bronze, silver, gold
    CUSTOMER_TYPE = np.random.randint(0, 3, user_samples)  # mobile, fixed, convergent
    HARDWARE_TYPE = np.random.randint(0, 5,
                                      user_samples)  # very cheap, cheap, mediocre, expensive, very expensive

    # activities 
    CUSTOMER_ID = np.random.randint(0, user_samples, activities_samples)
    PART_OF_DAY = np.random.randint(0, 3, activities_samples)  # morning, day, night
    DURATION = np.abs(
        np.array(300 * np.random.randn(activities_samples) + 600, dtype=np.int32))  # seconds
    USAGE_TYPE = np.random.randint(0, 4, activities_samples)  # calls, sms, data, mms
    MEASURE = np.random.randint(1, 1000, activities_samples)  # number of sms, mms, calls,
    ORIGINATOR_SERVICE_TYPE = np.random.randint(0, 2, activities_samples)  # mobile, fixed
    ORIGINATOR_LAT = np.abs(0.8 * np.random.randn(activities_samples) + 44)
    ORIGINATOR_LON = 1 * np.abs(np.random.randn(activities_samples) + 16)
    RECEIVER_SERVICE_TYPE = np.random.randint(0, 2, activities_samples)  # mobile, fixed
    RECEIVER_LAT = np.abs(0.8 * np.random.randn(activities_samples) + 44)
    RECEIVER_LON = 1 * np.abs(np.random.randn(activities_samples) + 16)

    users_table = {
        'USER_ID': pd.Series(USER_ID),
        'AGE': pd.Series(AGE),
        'SEX': pd.Series(SEX),
        'TENURE': pd.Series(TENURE),
        'USER_TYPE': pd.Series(USER_TYPE),
        'LAT': pd.Series(LAT),
        'LON': pd.Series(LON),
        'REVENUE': pd.Series(REVENUE),
        'CUSTOMER_TYPE': pd.Series(CUSTOMER_TYPE),
        'HARDWARE_TYPE': pd.Series(HARDWARE_TYPE),
    }

    activities_table = {
        'CUSTOMER_ID': pd.Series(CUSTOMER_ID),
        'PART_OF_DAY': pd.Series(PART_OF_DAY),
        'DURATION': pd.Series(DURATION),
        'USAGE_TYPE': pd.Series(USAGE_TYPE),
        'MEASURE': pd.Series(MEASURE),
        'ORIGINATOR_SERVICE_TYPE': pd.Series(ORIGINATOR_SERVICE_TYPE),
        'ORIGINATOR_LAT': pd.Series(ORIGINATOR_LAT),
        'ORIGINATOR_LON': pd.Series(ORIGINATOR_LON),
        'RECEIVER_SERVICE_TYPE': pd.Series(RECEIVER_SERVICE_TYPE),
        'RECEIVER_LAT': pd.Series(RECEIVER_LAT),
        'RECEIVER_LON': pd.Series(RECEIVER_LON),
    }

    users_frame = pd.DataFrame(users_table)
    activities_frame = pd.DataFrame(activities_table)
    return (users_frame, activities_frame)


def create_plan_dataset(user_type):
    samples = 1000
    # SMS, MMS, CALLS, NET
    user_types = []
    for type in range(user_type):
        activities = np.random.randn(samples, 4) * 2 + 50
        if type == 0:
            activities[:, 0] *= 10
        elif type == 1:
            activities[:, 1] *= 10
        elif type == 2:
            activities[:, 2] *= 10
        else:
            activities[:, 3] *= 10
        user_types.append(activities)

    return user_types


if __name__ == "__main__":
    x = create_plan_dataset(4)
    user_1 = x[3][0]
    predicted = plan_predictor.find_distance(sms_usage=user_1[0],
                                             mms_usage=user_1[1],
                                             call_usage=user_1[2],
                                             mbyte_usage=user_1[3],
                                             price=100)
    print(predicted)
