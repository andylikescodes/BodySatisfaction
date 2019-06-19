#!/bin/bash

python man.py

aws s3 cp ../outputs/man_aggregate.csv s3://body-satisfaction/man_aggregate.csv
aws s3 cp ../outputs/man_raw.csv s3://body-satisfaction/man_raw.csv

sudo shutdown -P now