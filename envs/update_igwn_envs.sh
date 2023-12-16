#!/bin/bash

# Get the current operating system and architecture
current_os=$(uname)
current_arch=$(uname -m)

current_conda="conda"
# Download the link based on the operating system and architecture
if [ "$current_os" = "Darwin" ]; then
 current_os="osx"
fi
if [ "$current_os" = "Linux" ]; then
 current_os="linux"
fi
if [ "$current_arch" = "x86_64" ]; then
 # Convert x86_64 to x64
 current_arch="64"
fi
if command -v mamba > /dev/null 2>&1; then
   # use mamba
   current_conda="mamba"
fi

$current_conda env update -f https://computing.docs.ligo.org/conda/environments/$current_os-$current_arch/igwn-py310.yaml --prune

