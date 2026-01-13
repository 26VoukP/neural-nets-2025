# NeuralNets2025

## Overview

NeuralNets2025 is a Java-based educational neural network library that implements a simple feedforward neural network with support for multiple activation functions, backpropagation, and customizable architecture. The project is designed for learning and experimentation with neural networks, including XOR and other basic supervised learning tasks.

**Note:** Versions for previous labs can be found in the commit history. Only demo configuration and data files (those starting with "demo") are tracked in the repository; other CSV and JSON files remain local only.

## Features

- Feedforward neural network with configurable number of layers
- **JSON-based configuration** - Edit network settings without recompiling
- Customizable number of layers and neurons per layer
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
│   └── NLayerNetwork.java      # Main neural network implementation
├── bin/                         # Compiled class files (auto-generated, ignored)
├── lib/
│   └── gson-2.10.1.jar         # JSON parsing library
├── demo-network-config.json     # Example network configuration file
├── demo-input-table.csv         # Example training input data
├── demo-truth-table.csv         # Example ground truth data
├── compile.bat                  # Compilation script (Windows)
├── run.bat                      # Execution script (Windows)
├── run-VoukAsus.bat             # Custom execution script
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

**Note:** Only demo CSV and JSON files (those starting with "demo") are tracked in the repository. Other CSV and JSON files are ignored and remain local only. Versions for previous labs can be found in the commit history.

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

# Linux/macOS
javac -cp "lib/gson-2.10.1.jar" -d bin src/ABCNetwork.java
```

**Run:**
```bash
# Windows
java -cp "bin;lib\gson-2.10.1.jar" NLayerNetwork

# Linux/macOS
java -cp "bin:lib/gson-2.10.1.jar" NLayerNetwork
```

## Configuration

The network is configured via a JSON configuration file (e.g., `demo-network-config.json`), which can be edited at runtime without recompiling. All settings are loaded when the program starts. You can specify a custom config file as a command-line argument: `java -cp "bin;lib\gson-2.10.1.jar" NLayerNetwork your-config.json`

**Note:** Only demo configuration files (those starting with "demo") are tracked in the repository. Other configuration files remain local only. Versions for previous labs can be found in the commit history.

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
| `numActivationLayers` | int | Number of activation layers (excluding input layer) |
| `layerSizes` | int[] | Array specifying the size of each layer, including input layer (e.g., `[2, 5, 3]` for 2 inputs, 5 hidden, 3 outputs) |
| `activationName` | string | Activation function: `"sigmoid"`, `"tanh"`, or `"linear"` |

#### `training` - Training Hyperparameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `learningRate` | double | Learning rate for gradient descent (e.g., 0.3) |
| `ECutoff` | double | Training stops when average error falls below this value |
| `IterationMax` | int | Maximum number of training epochs |
| `numCases` | int | Number of training examples |
| `keepAlive` | int | Print training progress every N iterations (0 to disable) |

#### `arrayParameters` - Data Loading and Weight Management
| Parameter | Type | Description |
|-----------|------|-------------|
| `min` | double | Minimum value for random weight initialization |
| `max` | double | Maximum value for random weight initialization |
| `loadWeightsFromFile` | boolean | Load weights from binary file instead of initializing |
| `saveWeightsToFile` | boolean | Save trained weights to binary file after training |
| `inputWeightsFileName` | string | Filename to load weights from |
| `outputWeightsFileName` | string | Filename to save weights to |
| `inputTableFileName` | string | CSV file containing training inputs (if specified, data will be loaded from CSV) |
| `truthTableFileName` | string | CSV file containing ground truth outputs (if specified, data will be loaded from CSV) |

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
    "numActivationLayers": 3,
    "layerSizes": [2, 5, 3],
    "activationName": "sigmoid"
  },
  "training": {
    "learningRate": 0.3,
    "ECutoff": 0.0002,
    "IterationMax": 100000,
    "numCases": 4,
    "keepAlive": 0
  },
  "arrayParameters": {
    "min": -1.5,
    "max": 1.5,
    "loadWeightsFromFile": false,
    "saveWeightsToFile": false,
    "inputWeightsFileName": "weights.bin",
    "outputWeightsFileName": "weights.bin",
    "inputTableFileName": "demo-input-table.csv",
    "truthTableFileName": "demo-truth-table.csv"
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

When `inputTableFileName` and `truthTableFileName` are specified in the configuration, the network loads data from CSV files.

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

- **Dimension validation**: The CSV must match the dimensions specified in the config (`numCases × layerSizes[0]` for inputs, `numCases × layerSizes[-1]` for outputs)
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
Layer sizes: [2, 5, 3]
Number of activation layers: 3
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
1. Create or edit a configuration file (e.g., `demo-network-config.json`) with desired architecture and training parameters
2. Set `"training": true` and `"loadWeightsFromFile": false`
3. Run `.\compile.bat` then `.\run.bat` (or specify your config file: `.\run.bat your-config.json`)

### Load and test existing weights
1. Set `"training": false` and `"loadWeightsFromFile": true`
2. Set `"runTestCases": true` to see results
3. Run the network

### Experiment with different learning rates
1. Edit `"learningRate"` in the config file
2. No need to recompile - just run again

### Use custom training data
1. Create CSV files (e.g., `input-table.csv` and `truth-table.csv`) with your data
2. Set `"loadTruthTableFromCSV": true` in your configuration file
3. Update `"numCases"`, `"numInputs"`, and `"numOutputs"` to match your data dimensions
4. Reference the CSV files in your config's `inputTableFileName` and `truthTableFileName` fields

**Note:** Only demo CSV files (those starting with "demo") are tracked in the repository. Your custom CSV files will remain local only.

## Dependencies

- **Java 21** or later
- **Gson 2.10.1** (included in `lib/` directory)

## Troubleshooting

### "package com.google.gson does not exist"
- **In VS Code**: The linter may show this error even though compilation works. This is a known issue with the Java Language Server and external JARs.
- **Solution**: Use the terminal scripts to compile and run, which handle the classpath correctly.

### "Could not find or load main class NLayerNetwork"
- **Windows**: Make sure you're using semicolon (`;`) in the classpath: `bin;lib\gson-2.10.1.jar`
- **Linux/Mac**: Make sure you're using colon (`:`) in the classpath: `bin:lib/gson-2.10.1.jar`
- **Git Bash on Windows**: Use the `.bat` files, not the `.sh` files

### CSV dimension mismatch errors
- Ensure the first line of your CSV contains the correct dimensions
- Verify that the number of rows/columns matches `numCases` and the corresponding layer sizes in `layerSizes` array

### Training not converging
- Try lowering the `learningRate`
- Increase `IterationMax`
- Adjust `ECutoff` to a higher value
- Try a different activation function

## Author

Vouk Praun-Petrovic

## License

This project is for educational and research purposes. No warranty is provided.
