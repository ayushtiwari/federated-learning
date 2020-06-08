import os
import pickle

import zmq
import tensorflow as tf
from tensorflow import keras

import numpy as np
import random
import dataset
import loader
import json

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
model = loader.cifar10()

# TEST_DATA_PATH='/root/mount/ayush/data/fashion_mnist/test'
TEST_DATA_PATH = '/root/mount/ayush/data/cifar10/test'

print("Loading test data...")
test_data = dataset.test_data(TEST_DATA_PATH)

x_test = test_data["x_test"]
y_test = test_data["y_test"]

print('x_test shape:', x_test.shape)
print(x_test.shape[0], 'test samples')

version = 0

NUM_CLIENTS = 20
NUM_ROUNDS = 10

SELECTOR_CONSTANT = 10


def register():
    print("Waiting for registrations...")
    for i in range(NUM_CLIENTS):
        reg_req = registrar.recv_pyobj()
        client_ip = reg_req["ip_addr"]
        clients[i] = client_ip

        print("Registering %s" % reg_req["ip_addr"])
        registrar.send_pyobj({"client_id": i})
        print("Registered %d/%d" % (i, NUM_CLIENTS))


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
            "epochs": 1,
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
        return {"success": False}

    print("Waiting for updates...")
    updates = []

    global version

    selected_clients_copy = selected_clients.copy()
    while len(selected_clients_copy) > 0:
        update = aggregator.recv_pyobj()

        print("Recieved update %d/%d" % (len(selected_clients_copy), len(selected_clients)))

        client_id = update["client_id"]
        model_version = update["version"]
        client_metrics = update["metrics"]

        if client_id in selected_clients_copy and model_version == version:
            print("Received update on version %s from client %s" % (model_version, client_id))
            print("Metrics: %s" % json.dumps(client_metrics, indent=4, sort_keys=True))
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

    version += 1
    print("Current version: %s" % version)
    print("Evaluating...")
    history = model.evaluate(x=x_test, y=y_test, batch_size=32)
    return {"success": True, "loss": history[0], "accuracy": history[1]}


print("Server started")
print("NUM_CLIENTS = %d" % NUM_CLIENTS)

register()

training_round = 0

log = []

while True:
    select()
    respond()
    server_metrics = aggregate()
    log.append({
        training_round: server_metrics
    })

    with open("logs/server/accuracy.json", 'w+') as file:
        json.dump(log, file, sort_keys=True, indent=4)

    training_round += 1
