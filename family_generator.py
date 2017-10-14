import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


feature_generator_parameters_type_0 = [ # Family
    # Date time measures mean and std for mean and for std. [0 - 1]
    0.7,   0.2,     0.002,  0.0009,
    0.79,  0.15,   0.0015, 0.00001,
    0.789, 0.2,    0.002,  0.002,
    0.71,  0.09,   0.0001, 0.0015,
    0.73,  0.12,   0.0025, 0.0001,
    0.72,  0.1,   0.001,  0.002,
    0.65,  0,         0.002,  0.0013,

    # Minutes parameters [3, 30 * 60] in seconds
    1*60,      20,    0.002,  0.0009,
    1*60,      8,    0.0015, 0.00001,
    1*60 + 20, 2,    0.002,  0.002,
    1*60 + 45, 5,    0.0001, 0.0015,
    1*60 - 10, 15,    0.0025, 0.0001,
    1*60 + 21, 10,    0.001,  0.002,
    20,        10,    0.002,  0.0013,

    # Usage type
    1, 1,

    # Number of location overlapping [0-1] percent of all calls
    0.5, 0.0015,

    # IOU when calls are not in the same repeater
    0.1, 0.0001
]

feature_generator_parameters_type_1 = [ # Friend
    # Date time measures mean and std for mean and for std. [0 - 1]
    0.7,   0.02,     0.002,  0.0009,
    0.79,  0.0015,   0.0015, 0.00001,
    0.89, 0.002,    0.002,  0.002,
    0.71,  0.0009,   0.0001, 0.0015,
    0.83,  0.0012,   0.0025, 0.0001,
    0.92,  0.001,   0.001,  0.002,
    0.15,  0.01,         0.002,  0.0013,

    # Minutes parameters [3, 30 * 60] in seconds
    3*60,      20,    0.002,  0.0009,
    1*60,      20,    0.0015, 0.00001,
    1*60 + 20, 10,    0.002,  0.002,
    1*60 + 45, 5,    0.0001, 0.0015,
    1*60 - 10, 10,    0.0025, 0.0001,
    1*60 + 21, 5,    0.001,  0.002,
    10,        10,    0.002,  0.0013,

    # Usage type
    1, 1,

    # Number of location overlapping [0-1] percent of all calls
    0.3, 0.0015,

    # IOU when calls are not in the same repeater
    0.2, 0.0001
]

feature_generator_parameters_type_2 = [ # Bussines And Other
        # Date time measures mean and std for mean and for std. [0 - 1]
    0.7,   0.02,     0.002,  0.0009,
    0.79,  0.0015,   0.0015, 0.00001,
    0.89, 0.002,    0.002,  0.002,
    0.71,  0.0009,   0.0001, 0.0015,
    0.83,  0.0012,   0.0025, 0.0001,
    0.92,  0.001,   0.001,  0.002,
    0.15,  0.01,         0.002,  0.0013,

    # Minutes parameters [3, 30 * 60] in seconds
    4*60,      20,    0.002,  0.0009,
    3*60,      20,    0.0015, 0.00001,
    2*60 + 20, 10,    0.002,  0.002,
    3*60 + 45, 5,    0.0001, 0.0015,
    60 - 10, 10,    0.0025, 0.0001,
    30 + 21, 5,    0.001,  0.002,
    1,        10,    0.002,  0.0013,

    # Usage type
    1, 1,

    # Number of location overlapping [0-1] percent of all calls
    0.8, 0.0015,

    # IOU when calls are not in the same repeater
    0.7, 0.0001
]

def choose_feature_generator_parameters(type):
    if type == 0:
        return feature_generator_parameters_type_0, 1
    elif type == 1:
        return feature_generator_parameters_type_1, 0
    else:
        return feature_generator_parameters_type_2, 0

def generate_dataset(num_samples, type):
    # Features: Date/time, Usage_type, Measure, service_type, origin_location,
    # receiving_service_type, receiving_location

    params, label = choose_feature_generator_parameters(type)

    date_time_features = []
    for i in range(0, 14):
        date_time_features.append(np.abs(params[2*i] + np.random.randn(num_samples) * params[2*i+1]))
    date_time_features = np.vstack(np.array(date_time_features))

    duration_features = []
    for i in range(0, 14):
        duration_features.append(np.abs(params[28 + 2*i] + np.random.randn(num_samples) * params[28 + 2*i + 1]))
    duration_features = np.vstack(np.array(duration_features))

    usage_type = np.array(np.abs(params[28] + np.random.randn(num_samples) * params[29]), dtype=np.int32)
    location_overlap = np.array(np.abs(params[30] + np.random.randn(num_samples) * params[31]), dtype=np.int32)
    iou = np.array(np.abs(params[32] + np.random.randn(num_samples) * params[33]), dtype=np.float32)

    rest_features = np.vstack((usage_type, location_overlap, iou))

    features = np.vstack((date_time_features, duration_features, rest_features))
    labels = np.repeat(label, features.shape[1])
    return features.T, labels

    #if type == 0:
    #    # Generate non-family
    #    monday = np.abs(12 + np.random.randn(num_samples) * 1.5)
    #    tuesday = np.abs(15 + np.random.randn(num_samples) * 0.5)
    #    wednesday = np.abs(14 + np.random.randn(num_samples) * 3)
    #    thursday = np.abs(17 + np.ran0dom.randn(num_samples) * 3.5)
    #    friday = np.abs(18 + np.random.randn(num_samples) * 0.1)
    #    saturday = np.zeros(num_samples)
    #    sunday = np.abs(5 + np.random.randn(num_samples) * 8)
#
    #    monday_measure = np.array(np.abs(20 + np.random.randn(num_samples) * 1), dtype=np.int32)
    #    tuesday_measure = np.array(np.abs(7 + np.random.randn(num_samples) * 1), dtype=np.int32)
    #    wednesday_measure = np.array(np.abs(8 + np.random.randn(num_samples) * 1), dtype=np.int32)
    #    thursday_measure = np.array(np.abs(10 + np.random.randn(num_samples) * 1), dtype=np.int32)
    #    friday_measure = np.array(np.abs(15 + np.random.randn(num_samples) * 1), dtype=np.int32)
    #    saturday_measure = np.zeros(num_samples)
    #    sunday_measure = np.array(np.abs(1 + np.random.randn(num_samples) * 1), dtype=np.int32)
#
    #    usage_type = np.array(np.abs(1 + np.random.randn(num_samples) * 1), dtype=np.int32)
    #    measure = np.array(np.abs(25 + np.random.randn(num_samples) * 4),   dtype=np.int32)
    #    service_type = np.random.randint(0, 2, (num_samples))
    #    origin_location = np.array(np.abs(3 + np.random.randn(num_samples) * 8), dtype=np.int32)
    #    receiving_type = np.random.randint(0, 2, (num_samples))
    #    receiving_location = np.array(np.abs(6 + np.random.randn(num_samples) * 4), dtype=np.int32)
    #    label = np.ones(num_samples)
#
    #else:
    #    # Generate family
    #    monday = np.abs(16 + np.random.randn(num_samples) * 1.5)
    #    tuesday = np.abs(11 + np.random.randn(num_samples) * 0.5)
    #    wednesday = np.abs(20 + np.random.randn(num_samples) * 3)
    #    thursday = np.abs(10 + np.random.randn(num_samples) * 3.5)
    #    friday = np.abs(9 + np.random.randn(num_samples) * 0.1)
    #    saturday = np.zeros(num_samples)
    #    sunday = np.abs(11 + np.random.randn(num_samples) * 8)
#
    #    monday_measure = np.array(np.abs(40 + np.random.randn(num_samples) * 1),
    #                              dtype=np.int32)
    #    tuesday_measure = np.array(np.abs(22 + np.random.randn(num_samples) * 1),
    #                               dtype=np.int32)
    #    wednesday_measure = np.array(np.abs(11 + np.random.randn(num_samples) * 1),
    #                                 dtype=np.int32)
    #    thursday_measure = np.array(np.abs(8 + np.random.randn(num_samples) * 1),
    #                                dtype=np.int32)
    #    friday_measure = np.array(np.abs(12 + np.random.randn(num_samples) * 1),
    #                              dtype=np.int32)
    #    saturday_measure = np.zeros(num_samples)
    #    sunday_measure = np.array(np.abs(1 + np.random.randn(num_samples) * 1), dtype=np.int32)
#
    #    usage_type = np.array(np.abs(1 + np.random.randn(num_samples) * 1), dtype=np.int32)
    #    measure = np.array(np.abs(50 + np.random.randn(num_samples) * 4), dtype=np.int32)
    #    service_type = np.random.randint(0, 2, (num_samples))
    #    origin_location = np.array(np.abs(4 + np.random.randn(num_samples) * 8),
    #                               dtype=np.int32)
    #    receiving_type = np.random.randint(0, 2, (num_samples))
    #    receiving_location = np.array(np.abs(7 + np.random.randn(num_samples) * 4),
    #                                  dtype=np.int32)
    #    label = np.zeros(num_samples)
#
    #features = np.vstack((monday, tuesday, wednesday, thursday, friday, saturday, sunday,
    #                      monday_measure, tuesday_measure, wednesday_measure,
    #                      thursday_measure, friday_measure, saturday_measure, sunday_measure,
    #                      usage_type, measure, service_type, origin_location,
    #                      receiving_location, receiving_type)).T
    #return features, label


family, fam_labels = generate_dataset(100, 0)
non_family, labels = generate_dataset(100, 1)

x = np.vstack((family, non_family))
y = np.hstack((fam_labels, labels))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = svm.SVC()
clf.fit(x_train, y_train)

print(accuracy_score(y_test, clf.predict(x_test)))
print(f1_score(y_test, clf.predict(x_test)))