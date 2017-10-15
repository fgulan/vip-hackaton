import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(1337)


feature_generator_parameters_type_0 = [ # Family
    # Date time measures mean and std for mean and for std. [0 - 1]
    0.7,   0.2,     0.002,  0.0009,
    0.79,  0.15,    0.0015, 0.00001,
    0.789, 0.2,     0.002,  0.002,
    0.71,  0.09,    0.0001, 0.0015,
    0.73,  0.12,    0.0025, 0.0001,
    0.72,  0.1,     0.001,  0.002,
    0.65,  0,       0.002,  0.0013,

    # Minutes parameters [3, 30 * 60] in seconds
    1*60,      20,    0.002,  0.0009,
    1*60,      8,     0.0015, 0.00001,
    1*60 + 20, 2,     0.002,  0.002,
    1*60 + 45, 5,     0.0001, 0.0015,
    1*60 - 10, 15,    0.0025, 0.0001,
    1*60 + 21, 10,    0.001,  0.002,
    20,        10,    0.002,  0.0013,

    # Usage type
    1, 1
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
    1, 1
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
    1, 1
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

    features = np.vstack((date_time_features, duration_features, usage_type))
    labels = np.repeat([label], features.shape[1])
    return features.T, labels

if __name__ == "__main__":
    family, fam_labels = generate_dataset(100, 0)
    non_family, labels = generate_dataset(100, 1)

    x = np.vstack((family, non_family))
    y = np.hstack((fam_labels, labels))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    print(accuracy_score(y_test, clf.predict(x_test)))
>>>>>>> b5bdbe5d8a66ce21da291176e9a2299efa541e1a
