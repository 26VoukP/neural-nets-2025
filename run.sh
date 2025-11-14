#!/bin/bash

ARG=${1:-}

CP="lib/gson-2.10.1.jar:bin"

if [[ -n "$ARG" ]]; then
  java -cp "$CP" ABCDNetwork "$ARG"
else
  java -cp "$CP" ABCDNetwork
fi

