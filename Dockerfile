FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get -y install libpython2.7 gringo curl
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN ln -s /lib/libclingo.so.3 /lib/libclingo.so.1

RUN mkdir /tmp/fastlas
WORKDIR /tmp/fastlas

COPY FastLAS ./FastLAS

RUN mv ./FastLAS /usr/local/bin
WORKDIR /
RUN rm -rf /tmp/fastlas

RUN curl -L -o opa https://openpolicyagent.org/downloads/v0.28.0/opa_linux_amd64
RUN chmod 755 ./opa
RUN mv ./opa /usr/local/bin

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
RUN install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

COPY ./generator /generator
