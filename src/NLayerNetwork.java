import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.math.BigDecimal;
import java.math.RoundingMode;

/*
* NLayerNetwork implements a configurable feedforward neural network with a configurable number of layers.
* It supports manual or random weight initialization, supervised training via backpropagation, and evaluation on
* arbitrary input tables. Activation function, learning rate, dataset source, and several runtime toggles are all
* driven through a JSON configuration file.
*
* Compilation and execution helpers:
* - Unix-like systems: run ./compile.sh then ./run.sh [optional-config-file]
* - Windows: run compile.bat then run.bat [optional-config-file]
* Both scripts expect Gson 2.10.1 in the local lib directory and emit class files into bin.
*
* External dependency: Gson (https://github.com/google/gson).
* Gson APIs used: com.google.gson.Gson#fromJson(Reader, Class) and com.google.gson.JsonObject getters.
*
* Author: Vouk Praun-Petrovic
* Created: September 9, 2024
*/
public class NLayerNetwork
{
   public double min, max;
   public double learningRate, ECutoff;
   public int numActivationLayers, numInputs, numOutputs;
   public int[] layerSizes;
   public int numCases, IterationMax, epochs, keepAlive;
   public double[][][] w;
   public double[][] a, inputTable, networkOutputs, groundTruths, psi, theta;
   public double[] targets;
   public double averageError;
   public String activationName;
   public boolean training, runTestCases, showInputs, showGroundTruths;
   public boolean loadWeightsFromFile, saveWeightsToFile;
   public String inputWeightsFileName, outputWeightsFileName, inputTableFileName, truthTableFileName;
   public String booleanOperation;
   public long trainStartTimeMillis, runStartTimeMillis, trainEndTimeMillis, runEndTimeMillis;
   public static final int INPUT_INDEX = 0;
   public static final int HIDDEN1_INDEX = 1;
   public static final int HIDDEN2_INDEX = 2;
   public String configFileName = "network-config.json";

/*
* Functional interface for activation functions and their derivatives.
*/
   @FunctionalInterface
   public interface Function
   {
      double apply(double x);
   }

   private static final java.util.HashMap<String, Function> ACTIVATION_MAP =
      new java.util.HashMap<>();
   static
   {
      ACTIVATION_MAP.put("sigmoid", x -> 1.0 / (1.0 + Math.exp(-x)));
      ACTIVATION_MAP.put("tanh", x -> 
         {
            double negFactor = (x < 0) ? 1.0 : -1.0;
            double exp = Math.exp(negFactor * 2.0 * x);
            return ((exp - 1.0) / (exp + 1.0)) * negFactor;
         });
      ACTIVATION_MAP.put("linear", x -> x);
   } // java.util.HashMap<String, Function> ACTIVATION_MAP


   private static final java.util.HashMap<String, Function> ACTIVATION_DERIVATIVE_MAP =
      new java.util.HashMap<>();
   static
   {
      ACTIVATION_DERIVATIVE_MAP.put("sigmoid", x -> 
         {
            double fx = ACTIVATION_MAP.get("sigmoid").apply(x);
            return fx * (1.0 - fx);
         });
      ACTIVATION_DERIVATIVE_MAP.put("tanh", x -> 
         {
            double fx = ACTIVATION_MAP.get("tanh").apply(x);
            return 1.0 - fx * fx;
         });
      ACTIVATION_DERIVATIVE_MAP.put("linear", x -> 1.0);
   } // java.util.HashMap<String, Function> ACTIVATION_DERIVATIVE_MAP
   public Function activationFunction, activationFunctionDerivative;

/*
* Initializes network parameters from a JSON configuration file.
* Reads network architecture, training parameters, weight settings, execution flags, and display preferences.
* If commandLineConfigFileName is null, the default configFileName field value is used.
* Throws java.io.IOException when the file cannot be read or required settings are missing.
*/
   public void initializeNetworkParams(String commandLineConfigFileName) throws java.io.IOException
   {
      try 
      {
         if (commandLineConfigFileName != null)
         {
            configFileName = commandLineConfigFileName;
         }

         JsonObject json = new Gson().fromJson(new java.io.FileReader(configFileName), JsonObject.class);
         
         String networkSectionHeader = "network";
         JsonObject network = getRequiredObject(json, networkSectionHeader); // Network architecture parameters
         numActivationLayers = getRequiredInt(network, "numActivationLayers", networkSectionHeader);
         layerSizes = getRequiredIntArray(network, "layerSizes", networkSectionHeader);
         if (layerSizes.length != numActivationLayers)
         {
            throw new java.io.IOException("Mismatch between number of activation layers and activation layer size list.");
         }
         numInputs = layerSizes[0]; // Named numInputs since it's used so often
         numOutputs = layerSizes[numActivationLayers - 1]; // Named numOutputs since it's used so often

         activationName = getRequiredString(network, "activationName", networkSectionHeader);
         activationFunction = ACTIVATION_MAP.get(activationName);
         activationFunctionDerivative = ACTIVATION_DERIVATIVE_MAP.get(activationName);

         String trainingSectionHeader = "training";
         JsonObject trainingParams = getRequiredObject(json, trainingSectionHeader); // Training parameters
         learningRate = getRequiredDouble(trainingParams, "learningRate", trainingSectionHeader);
         ECutoff = getRequiredDouble(trainingParams, "ECutoff", trainingSectionHeader);
         IterationMax = getRequiredInt(trainingParams, "IterationMax", trainingSectionHeader);
         numCases = getRequiredInt(trainingParams, "numCases", trainingSectionHeader);
         keepAlive = getRequiredInt(trainingParams, "keepAlive", trainingSectionHeader);

         String arraySectionHeader = "arrayParameters";
         JsonObject arrParams = getRequiredObject(json, arraySectionHeader); // Array initialization parameters
         min = getRequiredDouble(arrParams, "min", arraySectionHeader);
         max = getRequiredDouble(arrParams, "max", arraySectionHeader);
         loadWeightsFromFile = getRequiredBoolean(arrParams, "loadWeightsFromFile", arraySectionHeader);
         saveWeightsToFile = getRequiredBoolean(arrParams, "saveWeightsToFile", arraySectionHeader);
         inputWeightsFileName = getRequiredString(arrParams, "inputWeightsFileName", arraySectionHeader);
         outputWeightsFileName = getRequiredString(arrParams, "outputWeightsFileName", arraySectionHeader);
         inputTableFileName = getRequiredString(arrParams, "inputTableFileName", arraySectionHeader);
         truthTableFileName = getRequiredString(arrParams, "truthTableFileName", arraySectionHeader);

         String executionSectionHeader = "execution";
         JsonObject execution = getRequiredObject(json, executionSectionHeader); // Execution parameters
         training = getRequiredBoolean(execution, "training", executionSectionHeader);
         runTestCases = getRequiredBoolean(execution, "runTestCases", executionSectionHeader);
         booleanOperation = getRequiredString(execution, "booleanOperation", executionSectionHeader);

         String displaySectionHeader = "display";
         JsonObject display = getRequiredObject(json, displaySectionHeader); // Display parameters
         showInputs = getRequiredBoolean(display, "showInputs", displaySectionHeader);
         showGroundTruths = getRequiredBoolean(display, "showGroundTruths", displaySectionHeader);
      } // try()
      catch (java.io.IOException e) 
      {
         throw new java.io.IOException("Error reading config file: " + e.getMessage(), e);
      }
   } // initializeNetworkParams()

/*
* Retrieves a required JSON object from a parent JsonObject by key.
* Throws java.io.IOException when the requested section is absent.
*/
   private static JsonObject getRequiredObject(JsonObject parent, String key) throws java.io.IOException
   {
      if (parent.get(key) == null) 
      {
         throw new java.io.IOException("Missing required section '" + key + "' in config file");
      }
      return parent.getAsJsonObject(key);
   } // getRequiredObject(JsonObject parent, String key)

/*
* Reads a required integer value from the provided JsonObject.
* Includes the section name in the thrown java.io.IOException when the key is missing.
*/
   private static int getRequiredInt(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      return obj.get(key).getAsInt();
   } // getRequiredInt(JsonObject obj, String key, String section)

/*
* Reads a required double value from the provided JsonObject.
* Throws java.io.IOException with the section name when the key is absent.
*/
   private static double getRequiredDouble(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      return obj.get(key).getAsDouble();
   } // getRequiredDouble(JsonObject obj, String key, String section)

/*
* Reads a required string from the provided JsonObject.
* Throws java.io.IOException with the section name when the key is missing.
*/
   private static String getRequiredString(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      return obj.get(key).getAsString();
   } // getRequiredString(JsonObject obj, String key, String section)

/*
* Reads a required boolean from the provided JsonObject.
* Throws java.io.IOException with the section name when the key is missing.
*/
   private static boolean getRequiredBoolean(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      return obj.get(key).getAsBoolean();
   } // getRequiredBoolean(JsonObject obj, String key, String section)

/*
* Reads a required integer array from the provided JsonObject.
* Throws java.io.IOException with the section name when the key is missing or the array is invalid.
*/
   private static int[] getRequiredIntArray(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      JsonArray jsonArray = obj.get(key).getAsJsonArray();
      int[] result = new int[jsonArray.size()];
      for (int i = 0; i < jsonArray.size(); i++)
      {
         result[i] = jsonArray.get(i).getAsInt();
      }
      return result;
   } // getRequiredIntArray(JsonObject obj, String key, String section)

/*
* Allocates activations, weights, and optional buffers using the configured layer sizes and runtime flags.
* Call after initializeNetworkParams so dimensions and booleans are populated.
*/
   public void allocateNetworkArrays()
   {
      inputTable = new double[numCases][numInputs];
      networkOutputs = new double[numCases][numOutputs];

      w = new double[numActivationLayers - 1][][]; // w[0] is empty, w[1] operates on input layer, etc
      for (int n = 0; n < numActivationLayers - 1; n++)
      {
         w[n] = new double[layerSizes[n]][layerSizes[n + 1]]; // the n-index corresponds to the layer the weights output
      }
      a = new double[numActivationLayers][arrMax(layerSizes, 1, numActivationLayers)]; // inputs are assigned as pointer to a[0]

      if (training)
      {
         psi = new double[numActivationLayers][arrMax(layerSizes, 1, numActivationLayers)];  // one psi array for each layer
         theta = new double[numActivationLayers - 1][arrMax(layerSizes, 1, numActivationLayers - 1)]; // -1 since there's no output layer theta 
      }

      if (training | showGroundTruths)
      {
         groundTruths = new double[numCases][layerSizes[numActivationLayers - 1]];
      }
   } // allocateNetworkArrays()

/** 
* Finds the maximum value in an integer array between start and end indices.
* Returns the maximum value found.
*/
   public static int arrMax(int[] arr, int start, int end)
   {
      int max = Integer.MIN_VALUE;
      for (int index = start; index < end; index++)
      {
         if (arr[index] > max)
         {
            max = arr[index];
         }
      }
      return max;
   } // arrMax(int[] arr, int start, int end)

/*
* Populates network data structures by loading inputs/targets and initializing weights per configuration.
* Files are read when loadWeightsFromFile is true; otherwise random values are used.
* Throws java.io.IOException when required CSV or weight files are missing or malformed.
*/
   public void populateNetworkArrays() throws java.io.IOException
   {
      loadInputsAndTruthTable();

      if (loadWeightsFromFile)
      {
         loadWeightsFromFile();
      }
      else
      {
         generateRandomWeights();
      }
   } // populateNetworkArrays()

/*
* Loads the input and truth tables from CSV files specified in the configuration.
* Throws java.io.IOException when either file is missing, malformed, or has unexpected dimensions.
*/
   public void loadInputsAndTruthTable() throws java.io.IOException
   {
      loadDoubleArrfromCSV(inputTable, numCases, numInputs, inputTableFileName); // Load the input data
      loadDoubleArrfromCSV(groundTruths, numCases, numOutputs, truthTableFileName); // Load the truth data
   } // loadInputsAndTruthTable()

/*
* Loads a two-dimensional array from a CSV file whose first line declares rows,cols.
* Validates the declared dimensions against the destination array and parses each numeric entry.
* Throws java.io.IOException when the file is missing, the size mismatches, or parsing fails.
*/
   public static void loadDoubleArrfromCSV(double[][] arr, int arrRows, int arrCols, String filename) throws java.io.IOException
   {
      try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(filename)))
      {
         String line = reader.readLine(); // First line must provide the remaining matrix dimensions as "rows,cols"
         int numRows = Integer.parseInt(line.split(",")[0]);
         int numCols = Integer.parseInt(line.split(",")[1]);
         String[] values;

         if (numRows != arrRows || numCols != arrCols)
         {
            throw new java.io.IOException("Error: CSV file '" + filename + "' has invalid dimensions. Expected " + arrRows 
               + "x" + arrCols + " but found " + numRows + "x" + numCols + ".");
         } // if (numRows != arrRows || numCols != arrCols)

         for (int r = 0; r < numRows; r++)
         {
            line = reader.readLine();
            values = line.split(",");
            for (int c = 0; c < numCols; c++)
            {
               try
               {
                  arr[r][c] = Double.parseDouble(values[c].trim());
               }
               catch (NumberFormatException e)
               {
                  throw new java.io.IOException("Error: Invalid number '" + values[c] + "' at row " + r + ", column " + c);
               }
            } // for (int c = 0; c < numCols; c++)
         } // for (int r = 0; r < numRows; r++)
      }  // try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(filename)))
      catch (java.io.FileNotFoundException e)
      {
         throw new java.io.IOException("Error: " + filename + " file not found.");
      }
   } // loadDoubleArrfromCSV(double[][] arr, int arrRows, int arrCols, String filename)


/*
* Loads network weights from the configured binary file, validating dimensions at every layer boundary.
* Throws java.io.IOException when the file cannot be opened or the serialized layout does not match the network.
*/
   public void loadWeightsFromFile() throws java.io.IOException
   {
      try (
         java.io.DataInputStream in = new java.io.DataInputStream(
            new java.io.FileInputStream(inputWeightsFileName))
      )
      {
         int layerInputDim;
         int layerOutputDim;

         for (int n = 1; n < numActivationLayers; n++) // N starts at 1 less weight layer than activations
         {
            layerInputDim = in.readInt();
            layerOutputDim = in.readInt();

            if (layerInputDim != layerSizes[n - 1] || layerOutputDim != layerSizes[n])
            {
               throw new java.io.IOException("Error: File dimensions for layer " + n + " weights (" + layerInputDim
                   + "-" + layerOutputDim +") do not match network (" + layerSizes[n - 1] + "-" + layerSizes[n] + ")"); 
            }

            for (int input = 0; input < layerSizes[n - 1]; input++)
            {
               for (int output = 0; output < layerSizes[n]; output++)
               {
                  w[n - 1][input][output] = in.readDouble();
               }
            } // for (int input = 0; input < layerSizes[n - 1]; input++)
         } // for (int n = 1; n < numActivationLayers; n++)
      } // try()
      catch (java.io.IOException e) 
      {
         throw new java.io.IOException("Error opening file for loading weights: " + e.getMessage(), e);
      }
   } // loadWeightsFromFile()

/*
* Writes the current network weights to the configured binary file, prefixing each matrix with its dimensions.
* Throws java.io.IOException when the file cannot be created or written.
*/
   public void saveWeights() throws java.io.IOException
   {
      try (
         java.io.DataOutputStream out = new java.io.DataOutputStream(
            new java.io.FileOutputStream(outputWeightsFileName))
      ) 
      {
         for (int n = 1; n < numActivationLayers; n++)
         {
            out.writeInt(layerSizes[n - 1]); // saves dimensions of network to the file
            out.writeInt(layerSizes[n]);

            for (int m = 0; m < layerSizes[n - 1]; m++) 
            {
               for (int k = 0; k < layerSizes[n]; k++) 
               {
                  out.writeDouble(w[n - 1][m][k]);
               }
            } // for (int m = 0; m < layerSizes[n - 1]; m++)
         } // for (int n = 1; n < numActivationLayers; n++)
      } // try()
      catch (java.io.IOException e) 
      {
         throw new java.io.IOException("Error opening file for saving weights: " + e.getMessage(), e);
      }
   } // saveWeights()

/*
* Generates random weights for each layer within the configured [min, max) range.
* Assumes min < max; callers must ensure bounds are sensible before invocation.
*/
   public void generateRandomWeights()
   {
      for (int n = 1; n < numActivationLayers; n++)
      {
         for (int m = 0; m < layerSizes[n - 1]; m++)
         {
            for (int k = 0; k < layerSizes[n]; k++)
            {
               w[n - 1][m][k] = generateRandomDouble(min, max);
            } // for (int k = 0; k < layerSizes[n]; k++)
         } // for (int m = 0; m < layerSizes[n - 1]; m++)
      } // for (int n = 1; n < numActivationLayers; n++)
   } // generateRandomWeights()

/*
* Generates a random double within [min, max).
* Caller must guarantee min is less than max.
*/
   public static double generateRandomDouble(double min, double max)
   {
      return Math.random() * (max - min) + min;
   }

/*
* Performs a full forward pass during training through all hidden layers and the output layer, and returns half the squared error.
* Assumes a[INPUT_INDEX] and targets are populated for the current case.
*/
   public double runNetworkForTrain()
   {
      double sum, omega, currentError;
      for (int n = 1; n < numActivationLayers - 1; n++) // exclude last layer
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            sum = 0.0;
            for (int m = 0; m < layerSizes[n - 1]; m++)
            {
               sum += w[n - 1][m][k] * a[n - 1][m];
            }
            theta[n][k] = sum;
            a[n][k] = activationFunction.apply(sum);
         } // for (int k = 0; k < layerSizes[n]; k++)
      } // for (int n = 1; n < numActivationLayers - 1; n++)

      int n = numActivationLayers - 1; // initialize n to be the last activation layer for this loop

      currentError = 0.0;
      for (int i = 0; i < numOutputs; i++)
      {
         sum = 0.0;
         for (int j = 0; j < layerSizes[n - 1]; j++)
         {
            sum += w[n - 1][j][i] * a[n - 1][j];
         }
         a[n][i] = activationFunction.apply(sum);
         omega = targets[i] - a[n][i];
         psi[n][i] = omega * activationFunctionDerivative.apply(sum);
         currentError += omega * omega;
      } // for (int i = 0; i < numOutputs; i++)
      return currentError / 2.0;
   }

/*
* Performs a forward pass using the current values in a[INPUT_INDEX] as input activations.
* Useful for inference or evaluation after training.
*/
   public void run()
   {
      double sum;

      for (int n = 1; n < numActivationLayers; n++)
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            sum = 0.0;
            for (int m = 0; m < layerSizes[n - 1]; m++)
            {
               sum += w[n - 1][m][k] * a[n - 1][m];
            }
            a[n][k] = activationFunction.apply(sum);
         } // for (int k = 0; k < layerSizes[n]; k++)
      } // for (int n = 1; n < numActivationLayers; n++)
   } // run()

/*
* Runs the network on every configured case, capturing timing metrics and copying outputs for later reporting.
*/
   public void runAllCases()
   {
      runStartTimeMillis = System.currentTimeMillis();
      for (int t = 0; t < numCases; t++)
      {
         a[INPUT_INDEX] = inputTable[t];
         run();
         System.arraycopy(a[numActivationLayers - 1], 0, networkOutputs[t], 0, numOutputs); // Efficient array copy
      }
      runEndTimeMillis = System.currentTimeMillis();
   } // runAllCases()

/*
* Trains the network using the configured input and target tables until the error cutoff or iteration limit is reached.
* Updates averageError and epochs, and records wall-clock timing.
*/
   public void loopTraining()
   {
      double totalError;
      
      epochs = 0;
      trainStartTimeMillis = System.currentTimeMillis();
      averageError = Double.MAX_VALUE;
      while (epochs < IterationMax && averageError > ECutoff)
      {
         totalError = 0.0;
         for (int t = 0; t < numCases; t++)
         {
            a[INPUT_INDEX] = inputTable[t];
            targets = groundTruths[t];

            totalError += runNetworkForTrain();

            train();
         } // for (int t = 0; t < numCases; t++)

         averageError = totalError / (double) numCases;
         epochs++;

         if (keepAlive != 0 && epochs % keepAlive == 0)
         {
            System.out.printf("Iteration %d, Error = %f\n", epochs, averageError);
         }
      } // while (epochs < IterationMax && averageError > ECutoff)
      trainEndTimeMillis = System.currentTimeMillis();
   } // loopTraining()

/*
* Performs one backpropagation update using the derivatives captured in psi and theta.
* Requires runNetworkForTrain to have populated psi for the output layer before invocation.
*/
   public void train()
   {
      double omega;

      for (int n = numActivationLayers - 1; n > HIDDEN2_INDEX; n--)
      {
         for (int k = 0; k < layerSizes[n]; k++)
         {
            omega = 0.0;
            for (int m = 0; m < layerSizes[n - 1]; m++)
            {
               omega = psi[n][k] * w[n - 1][m][k];
               w[n - 1][m][k] += learningRate * a[n - 1][m] * psi[n][k];
            }
            psi[n - 1][k] = omega * activationFunctionDerivative.apply(theta[n - 1][k]);
         } // for (int k = 0; k < layerSizes[n]; k++)
      } // for (int n = numActivationLayers - 1; n > HIDDEN2_INDEX; n--)

      int n = HIDDEN1_INDEX;
      for (int m = 0; m < layerSizes[n]; m++)
      {
         omega = 0.0;
         for (int k = 0; k < layerSizes[n + 1]; k++)
         {
            omega += psi[n+1][k] * w[n][m][k];
            w[n][m][k] += learningRate * a[n][m] * psi[n+1][k];
         }
         
         psi[n][m] = omega * activationFunctionDerivative.apply(theta[n][m]);

         for (int x = 0; x < layerSizes[n - 1]; x++)
         {
            w[n-1][x][m] += learningRate * a[n-1][x] * psi[n][m];
         }
      } // for (int m = 0; m < layerSizes[n]; m++)
   } // train()

/*
* Formats the first len entries of a double array as [v1, v2, ...] using 4 decimal places.
* Caller must ensure len does not exceed arr.length.
*/
   public static String formatDoubleArr(double[] arr, int len)
   {
      String result = "[";
      for (int index = 0; index < len; index++)
      {
         result += new BigDecimal(arr[index]).setScale(4, RoundingMode.HALF_UP).toString();
         if (index < len - 1) // every element except last one is followed by a comma
         {
            result += ", ";
         }
      } // for (int index = 0; index < len; index++)
      return result + "]";
   } // formatDoubleArr(double[] arr, int len)
   
/**
* Prints the results of the training process, including convergence status and final error.
*/
   public void printTrainingResults()
   {
      System.out.println("Training Results:");

      System.out.println("Training Time: " + (trainEndTimeMillis - trainStartTimeMillis) + " milliseconds");
      if (epochs == IterationMax)
         System.out.print("Warning: Training did not converge to desired error value within " +
                            IterationMax + " iterations. Final error: ");
      
      else if (averageError <= ECutoff)
         System.out.print("Training converged successfully after " + epochs + " iterations. Final error: "); 
      System.out.printf("%.4g%n", averageError, "\n");
   } // printTrainingResults()

/**
* Prints run-time metrics along with outputs for each case, honoring {@code showInputs} and {@code showGroundTruths}.
*/
   public void printRunResults()
   {
      System.out.println("Run Results:");
      System.out.println("Run Time: " + (runEndTimeMillis - runStartTimeMillis) + " milliseconds");
      for (int t = 0; t < numCases; t++)
      {
         if (showInputs)
            System.out.print("Inputs: " + formatDoubleArr(inputTable[t], numInputs) + " ");

         if (showGroundTruths)
            System.out.print("Ground Truth: " + formatDoubleArr(groundTruths[t], numOutputs) + " ");

         System.out.println("Output: " + formatDoubleArr(networkOutputs[t], numOutputs));
      } // for (int t = 0; t < numCases; t++)
   } // printRunResults()

/**
* Prints the network parameters currently loaded in memory, including architecture, training, and display settings.
*/
   public void printNetworkParameters()
   {
      System.out.println("======== Network Parameters Loaded from " + configFileName + " ========");

      System.out.println("Network Architecture:");
      for (int n = 0; n < numActivationLayers; n++)
      {
         System.out.print(layerSizes[n] + "-");
      }
      System.out.println();
      System.out.println("Activation Function: " + activationName);

      if (training) 
      {
         System.out.println("\nTraining Configuration:");
         System.out.println("Learning Rate: " + learningRate);
         System.out.println("Number of Training Cases: " + numCases);
         System.out.println("Training Error Cutoff: " + ECutoff);
         System.out.println("Max Training Iterations: " + IterationMax);
      } // if (training)
      
      System.out.println("\nNetwork Array Initializations:");
      if (!loadWeightsFromFile)
      {
         System.out.println("Weight Initialization Range: [" + min + ", " + max + ")");
      }
      System.out.println("Load Weights from " + inputWeightsFileName + ": " + loadWeightsFromFile);
      System.out.println("Save Weights to " + outputWeightsFileName + ": " + saveWeightsToFile);
      System.out.println("Load Truth Table from " + truthTableFileName);
      System.out.println("Loading test cases from " + inputTableFileName);


      System.out.println("\nConfiguration Booleans:");
      System.out.println("Run Test Cases: " + runTestCases);
      System.out.println("Show Inputs: " + showInputs);
      System.out.println("Show Ground Truths: " + showGroundTruths + "\n");
   } // printNetworkParameters()

/**
* Main entry point for running the network.
* @param args Command-line arguments
*/
   public static void main(String[] args)
   {
      NLayerNetwork p = new NLayerNetwork();
      String configFileName = null;

      if (args.length > 0)
      {
         configFileName = args[0]; // Use the first command-line argument as the config file name
      }

      try
      {
         p.initializeNetworkParams(configFileName);
         p.printNetworkParameters();
         p.allocateNetworkArrays();

         p.populateNetworkArrays();

         if (p.training)
         {
            p.loopTraining();
            p.printTrainingResults();
         }

         if (!p.training || p.runTestCases)
         {
            p.runAllCases();
            p.printRunResults();
         }

         if (p.saveWeightsToFile)
         {
            p.saveWeights();
         }
      } // try
      catch (java.io.IOException | IllegalArgumentException e)
      {
         System.err.println(e.getMessage());
      }
   } // main(String[] args)
} // class NLayerNetwork
