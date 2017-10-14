import numpy as np

np.random.seed(1337)
# Current plans (SMS,MMS, CALLS, INTERNET, PRICE)
COMBINATIONS = np.array([[300, 20, 60, 10, 40],
                         [50, 400, 70, 40, 50],
                         [40, 20, 560, 90, 45],
                         [1, 100, 120, 490, 60]])


def find_distance(sms_usage, mms_usage, mbyte_usage, call_usage, price):
    # Filter all the prices which the user is
    filter_price = COMBINATIONS[COMBINATIONS[:, -1] <= price + 10]

    safe_margin = 10
    current_usage = np.array([sms_usage, mms_usage, call_usage, mbyte_usage]) + safe_margin
    min = 10000000
    recommended = None
    for plan in filter_price:
        diff = np.linalg.norm(current_usage - plan[:-1])
        if diff < min:
            min = diff
            recommended = plan
    return recommended, min
