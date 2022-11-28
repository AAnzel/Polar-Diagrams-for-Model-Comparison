#!/bin/bash

# STEP 1: Create the destination directory
mkdir -p Files

# STEP 2: Copy the script into the destination directory
cp Dataset_1_wget-20221011042255.sh ./Files/

# STEP 3: Move into the destination directory
cd ./Files

# STEP 4: Run the official script to download data
bash Dataset_1_wget-20221011042255.sh -s
