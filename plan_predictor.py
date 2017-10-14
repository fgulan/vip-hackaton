import numpy as np

np.random.seed(1337)
# Current plans (SMS, CALLS, INTERNET, PRICE)
COMBINATIONS = np.random.randint(20, 100, (100, 4))


def find_distance(sms_usage, mbyte_usage, call_usage, price):
    # Filter all the prices which the user is
    filter_price = COMBINATIONS[COMBINATIONS[:, -1] <= price + 10]

    safe_margin = 10
    current_usage = np.array([sms_usage, call_usage, mbyte_usage]) + safe_margin
    min = 10000000
    recommended = None
    for plan in filter_price:
        diff = np.linalg.norm(current_usage - plan[:-1])
        if diff < min:
            min = diff
            recommended = plan
    return recommended, min


sms = 5
mbyte = 50
call = 40
price = 50

print(find_distance(sms, mbyte, call, price))
