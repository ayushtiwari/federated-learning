# federated-learning

## Getting Started
Install dependencies
```
pip install -r requirements.txt
```
Create data partition
```
python scripts/data/cifar10.py <NUM_CLIENTS>
```
Update NUM_CLIENTS, SELECTOR_CONSTANT variable in server.py

Start Clients
```
docker-compose up --build --scale client=NUM_CLIENT
```
Start Server
```
python server.py
```
