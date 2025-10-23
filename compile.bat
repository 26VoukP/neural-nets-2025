@echo off
javac -cp "lib\gson-2.10.1.jar" -d bin src\ABCNetwork.java
if %errorlevel% == 0 (
    echo Compilation successful!
) else (
    echo Compilation failed!
)
