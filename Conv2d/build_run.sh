#!/bin/bash
nvcc main.cu -o app -lcudnn
clear
./app
