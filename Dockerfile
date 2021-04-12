FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get -y install python3 libpython2.7 wget gringo
RUN ln -s /lib/libclingo.so.3 /lib/libclingo.so.1

RUN mkdir /tmp/fastlas
WORKDIR /tmp/fastlas

COPY ./FastLAS .

RUN mv ./FastLAS /usr/local/bin
WORKDIR /
RUN rm -rf /tmp/fastlas

COPY ./generator /generator
