import pickle
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Saving client data")
for client_id in range(10):
    with open('/Users/ayushtiwari/Desktop/zmq/data/fashion_mnist/train/client_%s.pickle' % client_id, 'wb') as file:
        lower = 1000 * client_id
        upper = 1000 * (client_id + 1)
        pickle.dump({"x_train": x_train[lower:upper],
                     "y_train": y_train[lower:upper]}, file)

print("Saving server data")
with open('/Users/ayushtiwari/Desktop/zmq/data/fashion_mnist/test/server.pickle', 'wb') as file:
    pickle.dump({"x_test": x_test[0:1000],
                 "y_test": y_test[0:1000]}, file)
