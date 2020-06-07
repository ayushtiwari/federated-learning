# federated-learning

## Getting Started
Install dependencies
```
pip install -r requirements.txt
```

Update NUM_CLIENT variable in server.py

Start Clients
```
docker-compose up --build --scale client=NUM_CLIENT
```
Start Server
```
python server.py
```
