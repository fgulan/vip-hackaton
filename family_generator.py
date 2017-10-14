import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def generate_dataset(num_samples, type):
    # Features: Date/time, Usage_type, Measure, service_type, origin_location,
    # receiving_service_type, receiving_location

    if type == 0:
        # Generate non-family
        monday = np.abs(12 + np.random.randn(num_samples) * 1.5)
        tuesday = np.abs(15 + np.random.randn(num_samples) * 0.5)
        wednesday = np.abs(14 + np.random.randn(num_samples) * 3)
        thursday = np.abs(17 + np.random.randn(num_samples) * 3.5)
        friday = np.abs(18 + np.random.randn(num_samples) * 0.1)
        saturday = np.zeros(num_samples)
        sunday = np.abs(5 + np.random.randn(num_samples) * 8)

        monday_measure = np.array(np.abs(20 + np.random.randn(num_samples) * 1),
                                  dtype=np.int32)
        tuesday_measure = np.array(np.abs(7 + np.random.randn(num_samples) * 1),
                                   dtype=np.int32)
        wednesday_measure = np.array(np.abs(8 + np.random.randn(num_samples) * 1),
                                     dtype=np.int32)
        thursday_measure = np.array(np.abs(10 + np.random.randn(num_samples) * 1),
                                    dtype=np.int32)
        friday_measure = np.array(np.abs(15 + np.random.randn(num_samples) * 1),
                                  dtype=np.int32)
        saturday_measure = np.zeros(num_samples)
        sunday_measure = np.array(np.abs(1 + np.random.randn(num_samples) * 1), dtype=np.int32)

        usage_type = np.array(np.abs(1 + np.random.randn(num_samples) * 1), dtype=np.int32)
        measure = np.array(np.abs(25 + np.random.randn(num_samples) * 4), dtype=np.int32)
        service_type = np.random.randint(0, 2, (num_samples))
        origin_location = np.array(np.abs(3 + np.random.randn(num_samples) * 8),
                                   dtype=np.int32)
        receiving_type = np.random.randint(0, 2, (num_samples))
        receiving_location = np.array(np.abs(6 + np.random.randn(num_samples) * 4),
                                      dtype=np.int32)
        label = np.ones(num_samples)
    else:
        # Generate family
        monday = np.abs(16 + np.random.randn(num_samples) * 1.5)
        tuesday = np.abs(11 + np.random.randn(num_samples) * 0.5)
        wednesday = np.abs(20 + np.random.randn(num_samples) * 3)
        thursday = np.abs(10 + np.random.randn(num_samples) * 3.5)
        friday = np.abs(9 + np.random.randn(num_samples) * 0.1)
        saturday = np.zeros(num_samples)
        sunday = np.abs(11 + np.random.randn(num_samples) * 8)

        monday_measure = np.array(np.abs(40 + np.random.randn(num_samples) * 1),
                                  dtype=np.int32)
        tuesday_measure = np.array(np.abs(22 + np.random.randn(num_samples) * 1),
                                   dtype=np.int32)
        wednesday_measure = np.array(np.abs(11 + np.random.randn(num_samples) * 1),
                                     dtype=np.int32)
        thursday_measure = np.array(np.abs(8 + np.random.randn(num_samples) * 1),
                                    dtype=np.int32)
        friday_measure = np.array(np.abs(12 + np.random.randn(num_samples) * 1),
                                  dtype=np.int32)
        saturday_measure = np.zeros(num_samples)
        sunday_measure = np.array(np.abs(1 + np.random.randn(num_samples) * 1), dtype=np.int32)

        usage_type = np.array(np.abs(1 + np.random.randn(num_samples) * 1), dtype=np.int32)
        measure = np.array(np.abs(50 + np.random.randn(num_samples) * 4), dtype=np.int32)
        service_type = np.random.randint(0, 2, (num_samples))
        origin_location = np.array(np.abs(4 + np.random.randn(num_samples) * 8),
                                   dtype=np.int32)
        receiving_type = np.random.randint(0, 2, (num_samples))
        receiving_location = np.array(np.abs(7 + np.random.randn(num_samples) * 4),
                                      dtype=np.int32)
        label = np.zeros(num_samples)

    features = np.vstack((monday, tuesday, wednesday, thursday, friday, saturday, sunday,
                          monday_measure, tuesday_measure, wednesday_measure,
                          thursday_measure, friday_measure, saturday_measure, sunday_measure,
                          usage_type, measure, service_type, origin_location,
                          receiving_location, receiving_type)).T
    return features, label


family, fam_labels = generate_dataset(500, 0)
non_family, labels = generate_dataset(600, 1)

x = np.vstack((family, non_family))
y = np.hstack((fam_labels, labels))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = svm.SVC()
clf.fit(x_train, y_train)

print(f1_score(y_test, clf.predict(x_test)))
