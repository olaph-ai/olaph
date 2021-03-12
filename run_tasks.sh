#!/bin/bash

for filename in ../tasks/*.las; do
    echo "Learning $filename..."
    docker run -v $PWD/../tasks:/tasks fastlas FastLAS /$filename > ../models/$(basename "$filename" .las).lp
done
echo "Done"
