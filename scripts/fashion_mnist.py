import pickle
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Saving client data...")

train_path = '/root/mount/ayush/data/fashion_mnist/train'
for client_id in range(10):
    path = '%s/client_%s.pickle' % (train_path, client_id)
    with open(path, 'wb') as file:
        lower = 6000 * client_id
        upper = 6000 * (client_id + 1)
        pickle.dump({"x_train": x_train[lower:upper]/255.0,
                     "y_train": y_train[lower:upper]}, file)
    print("Saving Client %s train_data to %s" % (client_id, path))

test_path = '/root/mount/ayush/data/fashion_mnist/test'
print("Saving server data...")
path = '%s/server.pickle' % test_path
with open(path, 'wb') as file:
    pickle.dump({"x_test": x_test/255.0,
                 "y_test": y_test}, file)
    print("Saving server test data to %s" % (path))
