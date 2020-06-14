import json
import os
import random
import socket as sock
import time

import tensorflow as tf
import zmq
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

import dataset

context = zmq.Context()

# Registration Socket
registrar = context.socket(zmq.REQ)
registrar.connect("tcp://%s:5555" % (os.environ.get("SERVER_IP")))

print("tcp://%s:5555" % (os.environ.get("SERVER_IP")))

# Selection Socket
selector = context.socket(zmq.REQ)
selector.connect("tcp://%s:5556" % (os.environ.get("SERVER_IP")))

# Aggregation Socket
aggregator = context.socket(zmq.PUB)
aggregator.connect("tcp://%s:5557" % (os.environ.get("SERVER_IP")))

# Responder socket
responder = context.socket(zmq.REQ)
responder.connect("tcp://%s:5558" % (os.environ.get("SERVER_IP")))

ip_addr = sock.gethostbyname(sock.gethostname())


def register():
    print("Registering...")
    registrar.send_pyobj({"ip_addr": ip_addr})
    reply = registrar.recv_pyobj()
    return reply["client_id"]


def notify():
    print("Notifying selector...")
    selector.send_pyobj({"client_id": my_id})
    selector.recv_pyobj()


def request():
    print("Looking for requests...")
    responder.send_pyobj({"client_id": my_id})
    reply = responder.recv_pyobj()

    if reply["selected"]:
        print("Received request")
        model_dict = reply["model"]
        _model = tf.keras.models.model_from_json(model_dict["arch"])
        _model.set_weights(model_dict["weights"])
        _model.compile(optimizer=model_dict["optimizer"],
                       loss=model_dict["loss"],
                       metrics=model_dict["metrics_names"])
        _hparam = reply["hparam"]
        _version = reply["version"]
        return _model, _version, _hparam
    else:
        print("No request received")
        return None, None, None


def train():
    print("Computing update on version (%d)..." % version)

    history = model.fit(x_train, y_train,
                        batch_size=hparam["batch_size"],
                        epochs=hparam["epochs"],
                        workers=4)

    return history


def update():
    metrics = {
        "training": {
            "loss": history.history['loss'][-1],
            "accuracy": history.history['accuracy'][-1],
        }
    }

    logs.append({
        "version": version,
        "metrics": metrics
    })

    # print("Saving metrics...")
    # with open('%s/client_%s.json' % (os.environ.get("LOG_PATH"), my_id), 'w+') as file:
    #     json.dump(logs, file, indent=4, sort_keys=True)

    _update = dict({
        "client_id": my_id,
        "weights": model.get_weights(),
        "version": version,
        "points": x_train.shape[0],
        "metrics": metrics
    })

    print("Sending update...")
    aggregator.send_pyobj(_update)


print("[{}] Started".format(ip_addr))
my_id = register()
print("my_id: %s" % my_id)

print("Loading train data...")
train_data = dataset.train_data(os.environ["TRAIN_DATA_PATH"], my_id)

x_train = train_data["x_train"]
y_train = train_data["y_train"]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

logs = []

while True:
    notify()
    model, version, hparam = request()
    if model is None:
        continue

    history = train()
    update()
