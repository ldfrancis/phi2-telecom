#! /usr/bin/env bash

DOCUMENT_DIR="data/doc"

mkdir -p $DOCUMENT_DIR


# initialize states if it does not exist
if [ -f ".states" ]; then
    true
else
    touch .states
fi


# extract rel18 docs
if type "unrar" &> /dev/null
then
    if ! grep rel18 .states &> /dev/null
    then
    unrar x data/rel18.rar data/doc
    echo rel1878 >> .states
    echo "Done unpacking data/rel18.rar into data/doc/rel18"
    else
    echo "Unpacked data/rel18.rar into data/doc/rel18"
    fi
else
    echo "unrar command is not available."
    echo "Installing unrar ..."
    sudo apt update
    sudo apt install unrar
fi
