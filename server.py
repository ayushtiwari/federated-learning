import os
import pickle

import zmq
import tensorflow as tf
from tensorflow import keras

import numpy as np
import random

context = zmq.Context()

# Registration socket
registrar = context.socket(zmq.REP)
registrar.bind("tcp://*:5555")

# Selector Socket
selector = context.socket(zmq.REP)
selector.bind("tcp://*:5556")

# Aggregator Socket
aggregator = context.socket(zmq.SUB)
aggregator.bind("tcp://*:5557")
aggregator.setsockopt_string(zmq.SUBSCRIBE, "")

# Model Serving Socket
responder = context.socket(zmq.REP)
responder.bind("tcp://*:5558")

clients = {}
selected_clients = []
ready_clients = []

print("Loading model...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

TEST_DATA_PATH='/Users/ayushtiwari/Desktop/zmq/data/fashion_mnist/test'

print("Loading test data...")
with open('%s/server.pickle' % TEST_DATA_PATH, 'rb') as file:
    test_data = pickle.load(file)

x_test = test_data["x_test"]
y_test = test_data["y_test"]

print('x_test shape:', x_test.shape)
print(x_test.shape[0], 'test samples')

version = 0

NUM_CLIENTS = 1
NUM_ROUNDS = 10

SELECTOR_CONSTANT = 1


def register():
    print("Waiting for registrations...")
    for i in range(NUM_CLIENTS):
        reg_req = registrar.recv_pyobj()
        client_ip = reg_req["ip_addr"]
        clients[i] = client_ip

        print("Registering %s" % reg_req["ip_addr"])
        registrar.send_pyobj({"client_id": i})


def select():
    print("Selecting clients...")

    global selected_clients
    selected_clients = []

    global ready_clients
    ready_clients = []

    while True:

        if selector.poll(10000):
            request = selector.recv_pyobj(zmq.DONTWAIT)
            client_id = request["client_id"]
            print("Received request from %s" % client_id)
            ready_clients.append(client_id)
            selector.send_pyobj("")

            if len(ready_clients) == len(clients):
                break
        else:
            break

    if len(ready_clients) >= SELECTOR_CONSTANT:
        selected_clients = random.sample(ready_clients, SELECTOR_CONSTANT)

    print("Ready clients: %s" % ready_clients)
    print("Selected clients: %s" % selected_clients)
    return len(selected_clients)


def respond():
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

    reject_message = dict({
        "selected": False
    })

    ready_clients_copy = ready_clients.copy()
    while len(ready_clients_copy) > 0:
        request = responder.recv_pyobj()
        client_id = request["client_id"]
        if client_id in selected_clients:
            responder.send_pyobj(train_request)
        else:
            responder.send_pyobj(reject_message)

        ready_clients_copy.remove(client_id)


def aggregate():
    if len(selected_clients) <= 0:
        return

    print("Waiting for updates...")
    updates = []

    selected_clients_copy = selected_clients.copy()
    while len(selected_clients_copy) > 0:
        update = aggregator.recv_pyobj()
        client_id = update["client_id"]

        if client_id in selected_clients_copy:
            model_version = update["version"]
            print("Received update on version %s from client %s" % (model_version, client_id))
            updates.append(update)
            selected_clients_copy.remove(client_id)

    total_points = 0
    for update in updates:
        total_points += update["points"]

    print(total_points)

    weighted_avg = np.array([np.zeros(layer.shape) for layer in model.get_weights()])

    for update in updates:
        points = update["points"]
        weights = update["weights"]
        weighted_avg += (points / total_points) * np.array(weights)

    model.set_weights(weighted_avg.tolist())

    global version
    version += 1
    print("Current version: %s" % version)
    print("Evaluating...")
    model.evaluate(x=x_test, y=y_test, batch_size=32)


print("Server started")

register()

while True:
    select()
    respond()
    aggregate()
