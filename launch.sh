#!/bin/bash

rm /datadrive/samples/level_2/samples.zip
rm -r /datadrive/samples/level_2/smpls/
nohup python ~/create-song.py &
