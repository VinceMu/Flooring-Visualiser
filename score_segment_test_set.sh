#!/bin/bash

for test_file in ./data/segmentation_test_set/*; do
    echo "scoring: $test_file"
    python app.py --segment_image $test_file
done
