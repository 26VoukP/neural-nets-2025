#!/bin/bash
javac -cp "lib/gson-2.10.1.jar" -d bin src/ABCDNetwork.java
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
else
    echo "Compilation failed!"
fi

