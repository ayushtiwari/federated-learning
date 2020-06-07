import os

import zmq
import tensorflow as tf
from tensorflow import keras
import numpy as np

context = zmq.Context()

# Registration socket
registrar = context.socket(zmq.REP)
registrar.bind("tcp://*:5555")

# Model Serving Socket
selector = context.socket(zmq.REP)
selector.bind("tcp://*:5556")

# Aggregator Socket
aggregator = context.socket(zmq.SUB)
aggregator.bind("tcp://*:5557")
aggregator.setsockopt_string(zmq.SUBSCRIBE, "")

clients = {}
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
version = 0

NUM_CLIENTS = 2
NUM_ROUNDS = 10


def register():
    print("Regitering clients...")
    for i in range(NUM_CLIENTS):
        reg_req = registrar.recv_pyobj()
        client_ip = reg_req["ip_addr"]
        clients[i] = client_ip

        print("Registering %s" % reg_req["ip_addr"])
        registrar.send_pyobj({"client_id": i})


def select():
    print("Selecting clients...")
    train_request = dict({
        "selected": True,
        "model": dict({
            "arch": model.to_json(),
            "weights": model.get_weights(),
            "loss": model.loss,
            "optimizer": model.optimizer,
            "metrics_names": ['accuracy']
        }),
        "version": version,
        "hparam": dict({
            "epochs": 2,
            "batch_size": 32
        })
    })

    for _ in range(NUM_CLIENTS):
        request = selector.recv_pyobj()
        print("Sending model to %s" % request["client_id"])
        selector.send_pyobj(train_request)


def aggregate():
    print("Waiting for updates...")
    updates = []
    for i in range(NUM_CLIENTS):
        update = aggregator.recv_pyobj()
        print("Received update on version %s from client %s" % (update["version"], update["client_id"]))
        updates.append(update)

    total_points = 0
    for update in updates:
        total_points += update["points"]

    print(total_points)

    weighted_avg = np.array([np.zeros(layer.shape) for layer in model.get_weights()])

    for update in updates:
        points = update["points"]
        weights = update["weights"]
        weighted_avg += (points/total_points) * np.array(weights)

    model.set_weights(weighted_avg.tolist())

    global version
    version += 1
    print("Current version: %s" % version)
    print("Evaluating...")
    model.evaluate(x=x_test, y=y_test, batch_size=32)


print("Server started")

fashion_mnist = tf.keras.datasets.fashion_mnist
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test[0:1000]
y_test = y_test[0:1000]
x_test = x_test/255.0

register()

for _ in range(NUM_ROUNDS):
    select()
    aggregate()
