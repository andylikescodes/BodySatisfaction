#!/bin/bash

python woman.py

aws s3 cp ../outputs/woman_aggregate.csv s3://body-satisfaction/woman_aggregate.csv

aws s3 cp ../outputs/woman_raw.csv s3://body-satisfaction/woman_raw.csv

sudo shutdown -P now