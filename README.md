# NeuralNets2025

## Overview

NeuralNets2025 is a Java-based educational neural network library that implements a simple feedforward neural network with support for multiple activation functions, backpropagation, and customizable architecture. The project is designed for learning and experimentation with neural networks, including XOR and other basic supervised learning tasks.

## Features

- Feedforward neural network with one hidden layer
- **JSON-based configuration** - Edit network settings without recompiling
- Customizable number of input, hidden, and output neurons
- Multiple activation functions: sigmoid, tanh, linear
- Backpropagation training with mean squared error loss
- **CSV data loading** - Load training inputs and ground truth from CSV files
- **Binary weight persistence** - Save and load trained weights
- Comprehensive error handling with descriptive messages
- Cross-platform build scripts (Windows, Linux, macOS)

## Project Structure

```
NeuralNets2025/
├── src/
│   └── ABCNetwork.java         # Main neural network implementation
├── bin/                         # Compiled class files (auto-generated)
├── lib/
│   └── gson-2.10.1.jar         # JSON parsing library
├── network-config.json          # Network configuration file
├── input-table.csv              # Training input data (optional)
├── truth-table.csv              # Ground truth data (optional)
├── compile.bat / compile.sh     # Compilation scripts
├── run.bat / run.sh             # Execution scripts
└── README.md                    # This file
```

## Quick Start

### Compilation and Execution

Use the provided scripts for your operating system:

**Windows (PowerShell/CMD):**
```powershell
.\compile.bat    # Compile the code
.\run.bat        # Run the network
```

**Linux/macOS/Git Bash (on native Linux/Mac):**
```bash
chmod +x compile.sh run.sh    # Make executable (first time only)
./compile.sh                   # Compile the code
./run.sh                       # Run the network
```

**Note for Git Bash on Windows:** Use the `.bat` files even in Git Bash, as Windows Java requires Windows-style classpaths.

### Manual Compilation and Execution

If you prefer not to use the scripts:

**Compile:**
```bash
# Windows
javac -cp "lib\gson-2.10.1.jar" -d bin src\ABCNetwork.java
or 
compile

# Linux/macOS
javac -cp "lib/gson-2.10.1.jar" -d bin src/ABCNetwork.java
or 
chmod +x compile.sh run.sh
./compile.sh
```

**Run:**
```bash
# Windows
java -cp "bin;lib\gson-2.10.1.jar" ABCNetwork
or
run

# Linux/macOS
java -cp "bin:lib/gson-2.10.1.jar" ABCNetwork
or
./run.sh
```

## Configuration

The network is configured via `network-config.json`, which can be edited at runtime without recompiling. All settings are loaded when the program starts.

### Configuration File Structure

```json
{
  "network": { ... },
  "training": { ... },
  "arrayParameters": { ... },
  "execution": { ... },
  "display": { ... }
}
```

### Configuration Options

#### `network` - Network Architecture
| Parameter | Type | Description |
|-----------|------|-------------|
| `numInputs` | int | Number of input neurons |
| `numHidden` | int | Number of hidden layer neurons |
| `numOutputs` | int | Number of output neurons |
| `activationName` | string | Activation function: `"sigmoid"`, `"tanh"`, or `"linear"` |

#### `training` - Training Hyperparameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `learningRate` | double | Learning rate for gradient descent (e.g., 0.3) |
| `ECutoff` | double | Training stops when average error falls below this value |
| `IterationMax` | int | Maximum number of training epochs |
| `numCases` | int | Number of training examples |

#### `arrayParameters` - Data Loading and Weight Management
| Parameter | Type | Description |
|-----------|------|-------------|
| `min` | double | Minimum value for random weight initialization |
| `max` | double | Maximum value for random weight initialization |
| `manualWeights` | boolean | Use predefined weights (only for 2-2-1 networks) |
| `loadWeightsFromFile` | boolean | Load weights from binary file instead of initializing |
| `saveWeightsToFile` | boolean | Save trained weights to binary file after training |
| `inputWeightsFileName` | string | Filename to load weights from |
| `outputWeightsFileName` | string | Filename to save weights to |
| `loadTruthTableFromCSV` | boolean | Load training data from CSV files instead of hardcoded values |
| `inputTableFileName` | string | CSV file containing training inputs |
| `truthTableFileName` | string | CSV file containing ground truth outputs |

#### `execution` - Runtime Behavior
| Parameter | Type | Description |
|-----------|------|-------------|
| `training` | boolean | Enable training mode |
| `runTestCases` | boolean | Run test cases after training |
| `booleanOperation` | string | For predefined operations: `"OR"`, `"AND"`, `"XOR"`, or `"CUSTOM"` |

#### `display` - Output Options
| Parameter | Type | Description |
|-----------|------|-------------|
| `showInputs` | boolean | Display input values in results |
| `showGroundTruths` | boolean | Display ground truth values in results |

### Example Configuration

```json
{
  "network": {
    "numInputs": 2,
    "numHidden": 5,
    "numOutputs": 3,
    "activationName": "sigmoid"
  },
  "training": {
    "learningRate": 0.3,
    "ECutoff": 0.0002,
    "IterationMax": 100000,
    "numCases": 4
  },
  "arrayParameters": {
    "min": 0.1,
    "max": 1.5,
    "manualWeights": false,
    "loadWeightsFromFile": false,
    "saveWeightsToFile": true,
    "inputWeightsFileName": "nothing.bin",
    "outputWeightsFileName": "outputWeights.bin",
    "loadTruthTableFromCSV": true,
    "inputTableFileName": "input-table.csv",
    "truthTableFileName": "truth-table.csv"
  },
  "execution": {
    "training": true,
    "runTestCases": true,
    "booleanOperation": "CUSTOM"
  },
  "display": {
    "showInputs": true,
    "showGroundTruths": true
  }
}
```

## CSV Data Format

When `loadTruthTableFromCSV` is set to `true`, the network loads data from CSV files.

### Format Requirements

Both `input-table.csv` and `truth-table.csv` must follow this format:
```csv
rows,cols
value1,value2,...
value1,value2,...
```

**Example `input-table.csv` (4 cases, 2 inputs):**
```csv
4,2
0.0,0.0
0.0,1.0
1.0,0.0
1.0,1.0
```

**Example `truth-table.csv` (4 cases, 3 outputs):**
```csv
4,3
0.0,0.0,0.0
0.0,1.0,1.0
0.0,1.0,1.0
1.0,1.0,0.0
```

### CSV Loading Features

- **Dimension validation**: The CSV must match the dimensions specified in the config (`numCases × numInputs/numOutputs`)
- **Error reporting**: Provides specific error messages indicating which row/column has issues
- **Flexible values**: Supports any numeric values (not just 0.0 and 1.0)

## Weight Persistence

### Saving Weights

Set `saveWeightsToFile: true` in the config to save trained weights to a binary file after training completes.

### Loading Weights

Set `loadWeightsFromFile: true` to load previously saved weights instead of random initialization. This allows you to:
- Resume training from a checkpoint
- Use pre-trained weights for inference
- Skip training entirely if you have good weights

**Note:** The loaded weights file must match your current network architecture.

## Example Output

```
======== Network Parameters ========
Network Architecture:
2-5-3
Learning Rate: 0.3
Activation Function: sigmoid

Training Configuration:
Number of Training Cases: 4
Training Error Cutoff: 2.0E-4
Max Training Iterations: 100000

Training Results:
Training Time: 66 milliseconds
Training converged successfully after 41938 iterations. Final error: 1.9999934619648591E-4

Run Results:
Inputs: [0.0, 0.0] Ground Truth: [0.0, 0.0, 0.0] Output: [6.25560316315947E-4, 0.017526529885517905, 0.01829519728492776]
Inputs: [0.0, 1.0] Ground Truth: [0.0, 1.0, 1.0] Output: [0.011065486903209276, 0.9918755665338421, 0.9939092221337232]
Inputs: [1.0, 0.0] Ground Truth: [0.0, 1.0, 1.0] Output: [0.009013056397099355, 0.9881108028874507, 0.9842636471946032]
Inputs: [1.0, 1.0] Ground Truth: [1.0, 1.0, 0.0] Output: [0.9874990097876581, 0.9999447201786259, 0.010267187465203177]
```

## Common Tasks

### Train a new network
1. Edit `network-config.json` with desired architecture and training parameters
2. Set `"training": true` and `"loadWeightsFromFile": false`
3. Run `./compile.bat` (or `.sh`) then `./run.bat` (or `.sh`)

### Load and test existing weights
1. Set `"training": false` and `"loadWeightsFromFile": true`
2. Set `"runTestCases": true` to see results
3. Run the network

### Experiment with different learning rates
1. Edit `"learningRate"` in the config file
2. No need to recompile - just run again

### Use custom training data
1. Create `input-table.csv` and `truth-table.csv` with your data
2. Set `"loadTruthTableFromCSV": true`
3. Update `"numCases"`, `"numInputs"`, and `"numOutputs"` to match your data dimensions

## Dependencies

- **Java 21** or later
- **Gson 2.10.1** (included in `lib/` directory)

## Troubleshooting

### "package com.google.gson does not exist"
- **In VS Code**: The linter may show this error even though compilation works. This is a known issue with the Java Language Server and external JARs.
- **Solution**: Use the terminal scripts to compile and run, which handle the classpath correctly.

### "Could not find or load main class ABCNetwork"
- **Windows**: Make sure you're using semicolon (`;`) in the classpath: `bin;lib\gson-2.10.1.jar`
- **Linux/Mac**: Make sure you're using colon (`:`) in the classpath: `bin:lib/gson-2.10.1.jar`
- **Git Bash on Windows**: Use the `.bat` files, not the `.sh` files

### CSV dimension mismatch errors
- Ensure the first line of your CSV contains the correct dimensions
- Verify that the number of rows/columns matches `numCases`, `numInputs`, and `numOutputs` in the config

### Training not converging
- Try lowering the `learningRate`
- Increase `IterationMax`
- Adjust `ECutoff` to a higher value
- Try a different activation function

## Author

Vouk Praun-Petrovic

## License

This project is for educational and research purposes. No warranty is provided.
