import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10


NUM_CLIENTS=20

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# The data, split between train and test sets:
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

total_samples = x_train.shape[0]
samples_per_client = int(total_samples/NUM_CLIENTS)
print("Total Clients: %d" % NUM_CLIENTS)
print("Samples per client: %d" % samples_per_client)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("Saving client data...")
train_path = '/root/mount/ayush/data/cifar10/train'
for client_id in range(NUM_CLIENTS):
    path = '%s/client_%s.pickle' % (train_path, client_id)
    with open(path, 'wb') as file:
        lower = samples_per_client * client_id
        upper = samples_per_client * (client_id + 1)
        pickle.dump({"x_train": x_train[lower:upper]/255.0,
                     "y_train": y_train[lower:upper]}, file)
    print("Saving Client %s train_data to %s" % (client_id, path))

test_path = '/root/mount/ayush/data/cifar10/test'
print("Saving server data...")
path = '%s/server.pickle' % test_path
with open(path, 'wb') as file:
    pickle.dump({"x_test": x_test/255.0,
                 "y_test": y_test}, file)
    print("Saving server test data to %s" % path)
