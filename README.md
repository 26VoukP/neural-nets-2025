# NeuralNets2025

## Overview

NeuralNets2025 is a Java-based educational neural network library that implements a simple feedforward neural network (perceptron) with support for multiple activation functions, backpropagation, and customizable architecture. The project is designed for learning and experimentation with neural networks, including XOR and other basic supervised learning tasks.

## Features

- Feedforward neural network with one hidden layer
- Customizable number of input, hidden, and output neurons
- Supports manual and random weight initialization
- Multiple activation functions: sigmoid, relu, linear
- Backpropagation training with mean squared error loss
- Modular `FeedforwardLayer` class for easy extension
- Fully documented with JavaDoc

## Project Structure

- `src/` - Java source files
  - `Pereceptron.java`: Main neural network class, training loop, and demo
  - `FeedforwardLayer.java`: Represents a single layer in the network
- `bin/` - Compiled class files (ignored by git)
- `lib/` - (Optional) External libraries
- `README.md` - Project documentation
- `.gitignore` - Ignores IDE files, binaries, and temp files

## Usage

1. **Compile the project:**
	```
	javac src/*.java
	```
2. **Run the demo:**
	```
	java -cp src Pereceptron
	```
3. **Modify parameters:**
	- Edit `runNetworkWithHardcodedParams()` in `Pereceptron.java` to change network size, learning rate, activation function, or training mode.

## Example Output

```
Epoch: 1000, Average Error: 0.25
...
Final weights after training:
Weights from Input to Hidden Layer (w1):
[[0.5, -0.3], [0.2, 0.7], ...]
Weights from Hidden to Output Layer (w2):
[[1.1, -0.8, ...]]
```

## Author

Vouk Praun-Petrovic

## License

This project is for educational and research purposes. No warranty is provided.
## Getting Started

Welcome to the VS Code Java world. Here is a guideline to help you get started to write Java code in Visual Studio Code.

## Folder Structure

The workspace contains two folders by default, where:

- `src`: the folder to maintain sources
- `lib`: the folder to maintain dependencies

Meanwhile, the compiled output files will be generated in the `bin` folder by default.

> If you want to customize the folder structure, open `.vscode/settings.json` and update the related settings there.

## Dependency Management

The `JAVA PROJECTS` view allows you to manage your dependencies. More details can be found [here](https://github.com/microsoft/vscode-java-dependency#manage-dependencies).
