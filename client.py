import os

import zmq
import socket as sock

import tensorflow as tf
from tensorflow import keras

context = zmq.Context()

# Registration Socket
registrar = context.socket(zmq.REQ)
registrar.connect("tcp://%s:5555" % (os.environ.get("SERVER_IP")))

# Selection Socket
selector = context.socket(zmq.REQ)
selector.connect("tcp://%s:5556" % (os.environ.get("SERVER_IP")))

# Aggregation Socket
aggregator = context.socket(zmq.PUB)
aggregator.connect("tcp://%s:5557" % (os.environ.get("SERVER_IP")))

ip_addr = sock.gethostbyname(sock.gethostname())

NUM_POINTS = 1000


def register():
    print("Registering...")
    registrar.send_pyobj({"ip_addr": ip_addr})
    reply = registrar.recv_pyobj()
    return reply["client_id"]


def request():
    print("Looking for requests...")
    selector.send_pyobj({"client_id": my_id})
    reply = selector.recv_pyobj()

    if reply["selected"]:
        model_dict = reply["model"]
        _model = tf.keras.models.model_from_json(model_dict["arch"])
        _model.set_weights(model_dict["weights"])
        _model.compile(optimizer=model_dict["optimizer"],
                       loss=model_dict["loss"],
                       metrics=model_dict["metrics_names"])
        _hparam = reply["hparam"]
        _version = reply["version"]
        print(model_dict["loss"])
        return _model, _version, _hparam
    else:
        return None, None


def update():
    print("Sending update on version %s..." % version)
    aggregator.send_pyobj(dict({
        "client_id": my_id,
        "weights": model.get_weights(),
        "version": version,
        "points": NUM_POINTS
    }))


print("[{}] Started".format(ip_addr))
my_id = register()
print("my_id: %s" % my_id)

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (_, _) = fashion_mnist.load_data()
x_train = x_train[my_id * NUM_POINTS:(my_id + 1) * NUM_POINTS]
y_train = y_train[my_id * NUM_POINTS:(my_id + 1) * NUM_POINTS]
x_train = x_train / 255.0

while True:
    model, version, hparam = request()
    model.fit(x_train, y_train, epochs=hparam["epochs"], batch_size=hparam["batch_size"], validation_split=0.3)
    update()
