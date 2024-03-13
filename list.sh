#!/bin/bash

experiments_dir="./experiments"

if [ -d "$experiments_dir" ]; then
    echo "Listing folders inside $experiments_dir:"
    for folder in "$experiments_dir"/*; do
        if [ -d "$folder" ]; then
            folder_name=$(basename "$folder")
            if [[ ! $folder_name =~ ^_ ]]; then
                echo "$folder_name"
            fi
        fi
    done
else
    echo "Directory $experiments_dir does not exist."
fi