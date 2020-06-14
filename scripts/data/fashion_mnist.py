import pickle

from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

NUM_CLIENTS=1

num_classes = 10

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


total_samples = x_train.shape[0]
samples_per_client = int(total_samples/NUM_CLIENTS)
print("Total Clients: %d" % NUM_CLIENTS)
print("Samples per client: %d" % samples_per_client)


print("Saving client data...")
train_path = '/Users/ayushtiwari/Desktop/federated-learning/data/fashion_mnist/train'
for client_id in range(NUM_CLIENTS):
    path = '%s/client_%s.pickle' % (train_path, client_id)
    with open(path, 'wb') as file:
        lower = samples_per_client * client_id
        upper = samples_per_client * (client_id + 1)
        pickle.dump({"x_train": x_train[lower:upper],
                     "y_train": y_train[lower:upper]}, file)
    print("Saving Client %s train_data to %s" % (client_id, path))

test_path = '/Users/ayushtiwari/Desktop/federated-learning/data/fashion_mnist/test'
print("Saving server data...")
path = '%s/server.pickle' % test_path
with open(path, 'wb') as file:
    pickle.dump({"x_test": x_test,
                 "y_test": y_test}, file)
    print("Saving server test data to %s" % path)
