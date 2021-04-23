FROM python:3-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get -y install libpython2.7 gringo
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN ln -s /lib/libclingo.so.3 /lib/libclingo.so.1

RUN mkdir /tmp/fastlas
WORKDIR /tmp/fastlas

COPY FastLAS ./FastLAS

RUN mv ./FastLAS /usr/local/bin
WORKDIR /
RUN rm -rf /tmp/fastlas

COPY ./generator /generator
