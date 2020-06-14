import pickle


def train_data(train_data_path, client_id):
    with open('%s/client_%s.pickle' % (train_data_path, client_id), 'rb') as file:
        train_data = pickle.load(file)

    return train_data


def test_data(test_data_path):
    with open('%s/server.pickle' % test_data_path, 'rb') as file:
        test_data = pickle.load(file)

    return test_data
