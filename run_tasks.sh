#!/bin/bash

for filename in ../tasks/*.las; do
    echo "Learning $filename..."
    FastLAS --d $filename > ../models/$(basename "$filename" .las).lp
done
echo "Done"
