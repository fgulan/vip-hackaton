import numpy as np
import random as rnd
from sklearn.metrics import pairwise_distances

user_profile_generator = {
    'sex':  lambda: rnd.randint(0, 1),
    'age': lambda: rnd.randint(16, 80),
    'avg_sport_tv_duration_per_day': lambda: [0, rnd.randint(10, 120)][rnd.randint(0,1)],
    'avg_news_tv_duration_per_day': lambda: [0, rnd.randint(10, 120)][rnd.randint(0,1)],
    'avg_family_tv_duration_per_day': lambda: [0, rnd.randint(10, 120)][rnd.randint(0,1)],
    'avg_cartoon_tv_duration_per_day': lambda: [0, rnd.randint(10, 120)][rnd.randint(0,1)],

    'netflix_avg_duartion_mobile': lambda: [0, rnd.randint(90, 180)][abs(rnd.random()) > 0.9],
    'netflix_avg_duration_fixed': lambda: [0, rnd.randint(45, 180)] [abs(rnd.random()) > 0.65],
    'netflix_avg_part_of_day_mobile': lambda: [0.0, abs(rnd.random())][abs(rnd.random()) > 0.9],
    'netflix_avg_part_of_day_fixed': lambda: [0.0,  abs(rnd.random())][abs(rnd.random()) > 0.65],

    'yt_avg_duartion_mobile': lambda: [0, rnd.randint(1, 15)][abs(rnd.random()) > 0.5],
    'yt_avg_duration_fixed': lambda: [0, rnd.randint(3, 45)] [abs(rnd.random()) > 0.3],
    'yt_avg_part_of_day_mobile': lambda: [0.0, abs(rnd.random())] [abs(rnd.random()) > 0.5],
    'yt_avg_part_of_day_fixed': lambda: [0.0,  abs(rnd.random())]  [abs(rnd.random()) > 0.3],

    'fb_avg_duartion_mobile': lambda: [0, rnd.randint(5, 45)][abs(rnd.random()) > 0.3],
    'fb_avg_duration_fixed': lambda: [0, rnd.randint(5, 45)] [abs(rnd.random()) > 0.5],
    'fb_avg_part_of_day_mobile': lambda: [0.0, abs(rnd.random())][abs(rnd.random()) > 0.3],
    'fb_avg_part_of_day_fixed': lambda: [0.0,  abs(rnd.random())][abs(rnd.random()) > 0.5],

    'total_mobile_data_consumption_per_month': lambda: [rnd.random() * 0.5 + 3], # 3gb +- 0.1gb
    'total_fixed_data_consumption_per_month': lambda: [rnd.random() * 20 + 150],

    'total_mobile_minutes_spent': lambda: [int(rnd.random() * 30) + 400],
    'total_fixed_minutes_spent': lambda: [int(rnd.random() * 80) + 300],

    'total_number_of_sms_sent': lambda: [rnd.randint(0, 30)],

    'revenue_segment': lambda: [rnd.randint(0, 2)], # Bronze = 0, Silver = 1, Gold = 2

    'mobile_device': lambda: [rnd.randint(0, 10)], # Lets say there is 10 groups of mobile devices
}

def one_hot(size):
    arr = np.zeros(size)
    arr[rnd.randint(0, size-1)] = 1
    return arr

user_service_generator = {
    'fixed_data_speed': lambda: one_hot(4),
    'mobile_data_speed': lambda: one_hot(4),

    'fixed_data_size': lambda:  one_hot(5),
    'mobile_data_size': lambda: one_hot(5),

    'minutes_tarifa': lambda: one_hot(10),
    'sms_tarifa': lambda: one_hot(5),

    'mobile_device': lambda: one_hot(10),
    'hbo/netflix': lambda: np.array(rnd.randint(0, 1)),
    'social_network_options': lambda: np.array(rnd.randint(0,1)),
    'vip_now': lambda: np.array(rnd.randint(0, 1)),
    'tv_packets': lambda: np.array([
        rnd.randint(0,1), # Sport
        rnd.randint(0,1), # news
        rnd.randint(0,1), # porn
        rnd.randint(0,1)  # music
    ])
}

def load_user_data(num_of_samples):
    # Returns user_metrics_matrix and user_services_matrix

    # User metrics:

    # Age
    # Sex

    # Ganre tv avg day duration

    # netflix_avg_duration_mobile
    # netflix_avg_part_of_day_mobile
    # netflix_avg_duration_fixed
    # netflix_avg_part_of_day_fixed

    # youtube_avg_duration_mobile
    # youtube_avg_part_of_day_mobile
    # youtube_avg_duration_fixed
    # youtube_avg_part_of_day_fixed

    # fb_avg_duration_time_mobile
    # fb_avg_duration_time_fixed (fixed is via wifi router)
    # fb_avg_part_of_day_mobile
    # fb_avg_part_of_day_fixed

    # total mobile month data consumption
    # total fixed month data consumption

    # mobile device [iphone, samsung, ...]

    # revenue segment

    # User services:

    # fixed data packet [1G, 4G, 10G, flat]
    # fixed data speed [10mb/s, 20mb/s, 50mb/s]
    # mobile data packet []
    # mobile data speed []
    # minutes tarifes []
    # sms tarifes []
    # hbo/netflix
    # Mobilni uređaj [Iphone, Samsung, ....]
    # Smart Tv
    # Društvena Opcija
    # Vip Now
    # Nočna opcija
    # sport tv
    # news tv channels

    def generate_user_services():
        return np.hstack([gen() for name, gen in user_service_generator.items()])

    def generate_user_profile():
        return np.hstack([gen() for name, gen in user_profile_generator.items()])

    user_profiles = np.array([generate_user_profile() for _ in range(num_of_samples)])
    services = np.array([generate_user_services() for _ in range(num_of_samples)])

    return user_profiles, services

def create_similarity_matrix(user):
    return 1 - pairwise_distances(user, metric='cosine')


# From these create similarity matrix of NxF, NxM
user_feature_vectors, user_services = load_user_data(20)

# NxN, and remove similarities
user_similarty_matrix = create_similarity_matrix(user_feature_vectors) - np.eye(user_feature_vectors.shape[0])

# NxM
recommendations = np.matmul(user_similarty_matrix, user_services)

# Subtract recommendations with user_services because you do not need something that you already have.
normalized_recommendations = recommendations - user_services

recommended_services = np.argmax(normalized_recommendations, axis=1)

final_recommendations = [(index, normalized_recommendations[i][index]) for i, index in enumerate(recommended_services)]
for i, need in final_recommendations:
    print(i, need)