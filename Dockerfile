FROM python:3.6.10-slim

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  gcc

WORKDIR /app
RUN pip install pyzmq msgpack tensorflow

ADD . /app
