#!/bin/bash

# Download SAM 2.1 checkpoints

if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

echo "Downloading sam2.1_hiera_tiny.pt..."
$CMD "${BASE_URL}/sam2.1_hiera_tiny.pt" || { echo "Failed"; exit 1; }

echo "Downloading sam2.1_hiera_small.pt..."
$CMD "${BASE_URL}/sam2.1_hiera_small.pt" || { echo "Failed"; exit 1; }

echo "Downloading sam2.1_hiera_base_plus.pt..."
$CMD "${BASE_URL}/sam2.1_hiera_base_plus.pt" || { echo "Failed"; exit 1; }

echo "Downloading sam2.1_hiera_large.pt..."
$CMD "${BASE_URL}/sam2.1_hiera_large.pt" || { echo "Failed"; exit 1; }

echo "All checkpoints downloaded successfully."
