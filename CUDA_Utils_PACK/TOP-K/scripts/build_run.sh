#!/bin/bash

cd ../build
rm * -rf
cmake .. 
make
clear
./top_k
