FROM python:3-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get -y install libpython2.7
RUN apt-get -y install wget gringo
RUN ln -s /lib/libclingo.so.3 /lib/libclingo.so.1

RUN mkdir /tmp/ilasp
WORKDIR /tmp/ilasp

COPY ./FastLAS .

RUN mv ./FastLAS /usr/local/bin
WORKDIR /
RUN rm -rf /tmp/ilasp

COPY ./generator /generator
