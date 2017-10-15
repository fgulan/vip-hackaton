import numpy as np
from sklearn.metrics import pairwise_distances


def load_user_data():
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

    return np.array([
        [1, 2, 5, 6, 2, 1, 7],
        [2, 3, 1, 5, 2, 3, 5],
        [2, 4, 2, 5, 1, 3, 4]
    ]), np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1]
    ])


def create_similarity_matrix(user):
    return 1 - pairwise_distances(user, metric='cosine')


# From these create similarity matrix of NxF, NxM
user_feature_vectors, user_services = load_user_data()

# NxN, and remove similarities
user_similarty_matrix = create_similarity_matrix(user_feature_vectors) - np.eye(user_feature_vectors.shape[0])

# NxM
recommendations = user_similarty_matrix * user_services

# Subtract recommendations with user_services because you do not need something that you already have.
normalized_recommendations = recommendations - user_services

recommended_services = np.argmax(normalized_recommendations, axis=1)

final_recommendations = [(index, normalized_recommendations[i][index]) for i, index in enumerate(recommended_services)]
