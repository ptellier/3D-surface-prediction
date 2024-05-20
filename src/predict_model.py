import numpy as np
from train_model import get_classifier_data, train_surface_classifier

TESTING_IMG_IDS = [num for num in range(1, 20+1)]
TRAINING_IMAGE_IDS = [num for num in range(21, 26+1)]
MODE = 1


if __name__ == '__main__':
    X_train, y_train, num_neigh_train = get_classifier_data(TRAINING_IMAGE_IDS)
    X_test, y_test, num_neigh_test = get_classifier_data(TESTING_IMG_IDS)
    n_train = len(y_train)
    n_test = len(y_test)

    trained_surface_classifier = train_surface_classifier(X_train, y_train, num_neigh_train)
    train_predictions = trained_surface_classifier.predict(X_train)
    test_predictions = trained_surface_classifier.predict(X_test)

    train_accuracy = np.sum(train_predictions == y_train) / n_train
    test_accuracy = np.sum(test_predictions == y_test) / n_test
    mode_dummy_train_accuracy = np.sum((np.ones(n_train) * MODE) == y_train) / n_train
    mode_dummy_test_accuracy = np.sum((np.ones(n_test) * MODE) == y_test) / n_test
    print(f'train accuracy: {round(train_accuracy*100, 2)}%')
    print(f'test accuracy: {round(test_accuracy * 100, 2)}%')
    print(f'dummy train accuracy: {round(mode_dummy_train_accuracy * 100, 2)}%')
    print(f'dummy test accuracy: {round(mode_dummy_test_accuracy * 100, 2)}%')
