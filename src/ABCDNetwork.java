import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.math.BigDecimal;
import java.math.RoundingMode;

/*
* ABCDNetwork implements a configurable feedforward neural network with two hidden layers.
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
public class ABCDNetwork
{
   public double min, max;
   public double learningRate, ECutoff;
   public int numActivationLayers, numInputs, numHidden1, numHidden2, numOutputs;
   public int numCases, IterationMax, epochs;
   public double[][][] w;
   public double[][] a, inputTable, networkOutputs, groundTruths, psi, theta;
   public double[] targets;
   public double averageError;
   public String activationName;
   public boolean training, manualWeights, runTestCases, showInputs, showGroundTruths;
   public boolean loadWeightsFromFile, saveWeightsToFile, loadTruthTableFromCSV;
   public String inputWeightsFileName, outputWeightsFileName, inputTableFileName, truthTableFileName;
   public String booleanOperation;
   public long trainStartTimeMillis, runStartTimeMillis, trainEndTimeMillis, runEndTimeMillis;
   public static final int INPUT_INEDX = 0;
   public static final int HIDDEN1_INDEX = 1;
   public static final int HIDDEN2_INDEX = 2;
   public static final int OUTPUT_INDEX = 3;
   public static final String DEFAULT_CONFIG_FILE = "network-config.json";

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
   }


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
   }

   public Function activationFunction, activationFunctionDerivative;

/*
* Initializes network parameters from a JSON configuration file.
* Reads network architecture, training parameters, weight settings, execution flags, and display preferences.
* If configFileName is null, DEFAULT_CONFIG_FILE is used.
* Throws java.io.IOException when the file cannot be read or required settings are missing.
*/
   public void initializeNetworkParams(String configFileName) throws java.io.IOException
   {
      try 
      {
         numActivationLayers = 4; // Will later be read in from config file in this section when N-layer is implemented 

         JsonObject json = new Gson().fromJson(new java.io.FileReader(configFileName == null ? DEFAULT_CONFIG_FILE : configFileName), JsonObject.class);
         
         String networkSectionHeader = "network";
         JsonObject network = getRequiredObject(json, networkSectionHeader); // Network architecture parameters
         numInputs = getRequiredInt(network, "numInputs", networkSectionHeader);
         numHidden1 = getRequiredInt(network, "numHidden1", networkSectionHeader);
         numHidden2 = getRequiredInt(network, "numHidden2", networkSectionHeader);
         numOutputs = getRequiredInt(network, "numOutputs", networkSectionHeader);
         activationName = getRequiredString(network, "activationName", networkSectionHeader);
         activationFunction = ACTIVATION_MAP.get(activationName);
         activationFunctionDerivative = ACTIVATION_DERIVATIVE_MAP.get(activationName);

         String trainingSectionHeader = "training";
         JsonObject trainingParams = getRequiredObject(json, trainingSectionHeader); // Training parameters
         learningRate = getRequiredDouble(trainingParams, "learningRate", trainingSectionHeader);
         ECutoff = getRequiredDouble(trainingParams, "ECutoff", trainingSectionHeader);
         IterationMax = getRequiredInt(trainingParams, "IterationMax", trainingSectionHeader);
         numCases = getRequiredInt(trainingParams, "numCases", trainingSectionHeader);

         String arraySectionHeader = "arrayParameters";
         JsonObject arrParams = getRequiredObject(json, arraySectionHeader); // Array initialization parameters
         min = getRequiredDouble(arrParams, "min", arraySectionHeader);
         max = getRequiredDouble(arrParams, "max", arraySectionHeader);
         manualWeights = getRequiredBoolean(arrParams, "manualWeights", arraySectionHeader);
         loadWeightsFromFile = getRequiredBoolean(arrParams, "loadWeightsFromFile", arraySectionHeader);
         saveWeightsToFile = getRequiredBoolean(arrParams, "saveWeightsToFile", arraySectionHeader);
         inputWeightsFileName = getRequiredString(arrParams, "inputWeightsFileName", arraySectionHeader);
         outputWeightsFileName = getRequiredString(arrParams, "outputWeightsFileName", arraySectionHeader);
         loadTruthTableFromCSV = getRequiredBoolean(arrParams, "loadTruthTableFromCSV", arraySectionHeader);
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
* Allocates activations, weights, and optional buffers using the configured layer sizes and runtime flags.
* Call after initializeNetworkParams so dimensions and booleans are populated.
*/
   public void allocateNetworkArrays()
   {
      inputTable = new double[numCases][numInputs];
      networkOutputs = new double[numCases][numOutputs];

      w = new double[numActivationLayers - 1][][]; // -1 since there's no output layer weight array
      w[INPUT_INEDX] = new double[numInputs][numHidden1];
      w[HIDDEN1_INDEX] = new double[numHidden1][numHidden2];
      w[HIDDEN2_INDEX] = new double[numHidden2][numOutputs]; // the n-index corresponds to the layer the weights operate on

      a = new double[numActivationLayers][Math.max(Math.max(numHidden1, numHidden2), numOutputs)]; // inputs are assigned as pointer to a[0]

      if (training)
      {
         psi = new double[numActivationLayers][Math.max(numHidden1, Math.max(numHidden2, numOutputs))];  // one psi array for each layer
         theta = new double[numActivationLayers - 1][Math.max(numHidden1, numHidden2)]; // -1 since there's no output layer theta 
      }

      if (training | showGroundTruths)
      {
         groundTruths = new double[numCases][numOutputs];
      }
   } // allocateNetworkArrays()

/*
* Populates network data structures by loading inputs/targets and initializing weights per configuration.
* Files are read when loadTruthTableFromCSV or loadWeightsFromFile are true; otherwise defaults or random values are used.
* Throws java.io.IOException when required CSV or weight files are missing or malformed.
*/
   public void populateNetworkArrays() throws java.io.IOException
   {
      if (loadTruthTableFromCSV)
      {
         loadInputsAndTruthTable();
      }
      else
      {
         manuallySetInputsAndTruthTable();
      }

      if (loadWeightsFromFile)
      {
         loadWeightsFromFile();
      }
      else if (manualWeights) // Only a valid option for a  2-2-1 network 
      {
         setPredefinedWeights();
      }
      else
      {
         generateRandomWeights();
      }
   } // populateNetworkArrays()

/*
* Sets deterministic weights for the hand-crafted demonstration network.
* Only valid when the configured architecture matches the hard-coded dimensions.
*/
   public void setPredefinedWeights()
   {
      w[INPUT_INEDX][0][0] = 0.9404045278126735;
      w[INPUT_INEDX][0][1] = -1.0121547995672862;
      w[INPUT_INEDX][1][0] = 0.2241493210468354;
      w[INPUT_INEDX][1][1] = -1.432402348525032;

      w[HIDDEN1_INDEX][0][0] = 0.251086609938219;
      w[HIDDEN1_INDEX][1][0] = -0.41775164313179797;
   } // setPredefinedWeights()


/*
* Populates the built-in input table and truth table for the configured boolean operation.
* When booleanOperation is CUSTOM, fills the multi-output truth table; otherwise enforces single-output logic.
*/
   public void manuallySetInputsAndTruthTable()
   {
      inputTable[0][0] = 0.0;
      inputTable[0][1] = 0.0;
      inputTable[1][0] = 1.0;
      inputTable[1][1] = 0.0;
      inputTable[2][0] = 0.0;
      inputTable[2][1] = 1.0;
      inputTable[3][0] = 1.0;
      inputTable[3][1] = 1.0;

      if (!booleanOperation.equals("CUSTOM"))
      {
         if (training || showGroundTruths)
         {
            switch (booleanOperation)
            {
               case "OR" ->
               {
                  groundTruths[0][0] = 0.0;
                  groundTruths[1][0] = 1.0;
                  groundTruths[2][0] = 1.0;
                  groundTruths[3][0] = 1.0;
               }
               case "AND" ->
               {
                  groundTruths[0][0] = 0.0;
                  groundTruths[1][0] = 0.0;
                  groundTruths[2][0] = 0.0;
                  groundTruths[3][0] = 1.0;
               }
               case "XOR" ->
               {
                  groundTruths[0][0] = 0.0;
                  groundTruths[1][0] = 1.0;
                  groundTruths[2][0] = 1.0;
                  groundTruths[3][0] = 0.0;
               }
               default -> throw new IllegalArgumentException("Error populating truth table for '" + booleanOperation + "' operation.");
            } // switch (booleanOperation)
         } // if (training || showGroundTruths)
      } // if (!booleanOperation.equals("CUSTOM"))
      else
      {
         if (training || showGroundTruths)
         {
            groundTruths[0][0] = 0.0;  groundTruths[0][1] = 0.0;  groundTruths[0][2] = 0.0;
            groundTruths[1][0] = 0.0;  groundTruths[1][1] = 1.0;  groundTruths[1][2] = 1.0;
            groundTruths[2][0] = 0.0;  groundTruths[2][1] = 1.0;  groundTruths[2][2] = 1.0;
            groundTruths[3][0] = 1.0;  groundTruths[3][1] = 1.0;  groundTruths[3][2] = 0.0;
         }
      } // else
   } // manuallySetInputsAndTruthTable()

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

         int fileNumInputs = in.readInt();
         int fileNumHidden1 = in.readInt();
         
         if (fileNumInputs != numInputs || fileNumHidden1 != numHidden1) 
         {
            throw new java.io.IOException("Error: File dimensions for layer 1 weights (" + fileNumInputs + "-" + fileNumHidden1 +
                               ") do not match network (" + numInputs + "-" + numHidden1 + ")"); 
         }

         for (int m = 0; m < numInputs; m++) 
         {
            for (int k = 0; k < numHidden1; k++) 
            {
               w[INPUT_INEDX][m][k] = in.readDouble();
            }
         } // for (int m = 0; m < numInputs; m++)

         fileNumHidden1 = in.readInt();
         int fileNumHidden2 = in.readInt();

         if (fileNumHidden1 != numHidden1 || fileNumHidden2 != numHidden2) 
         {
            throw new java.io.IOException("Error: File dimensions for layer 2 weights (" + fileNumInputs + "-" + fileNumHidden1 +
                               ") do not match network (" + numInputs + "-" + numHidden1 + ")"); 
         }

         for (int k = 0; k < numHidden1; k++) 
         {
            for (int j = 0; j < numHidden2; j++) 
            {
               w[HIDDEN1_INDEX][k][j] = in.readDouble();
            }
         } // for (int k = 0; k < numHidden1; k++)

         fileNumHidden2 = in.readInt(); // Read and check dimensions for w[HIDDEN2_INDEX]
         int fileNumOutputs = in.readInt();

         if (fileNumHidden2 != numHidden2 || fileNumOutputs != numOutputs) 
         {
            throw new IndexOutOfBoundsException("Error: File dimensions for w[HIDDEN2_INDEX] (" + fileNumHidden2 + "-" + fileNumOutputs +
                               ") do not match network (" + numHidden2 + "-" + numOutputs + ")");
         }

         for (int j = 0; j < numHidden2; j++) 
         {
            for (int i = 0; i < numOutputs; i++) 
            {
               w[HIDDEN2_INDEX][j][i] = in.readDouble();
            }
         } // for (int j = 0; j < numHidden2; j++)
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
         out.writeInt(numInputs); // saves dimensions of network to the file
         out.writeInt(numHidden1);

         for (int m = 0; m < numInputs; m++) 
         {
            for (int k = 0; k < numHidden1; k++) 
            {
               out.writeDouble(w[INPUT_INEDX][m][k]);
            }
         } // for (int m = 0; m < numInputs; m++)

         out.writeInt(numHidden1);
         out.writeInt(numHidden2);

         for (int k = 0; k < numHidden1; k++) 
         {
            for (int j = 0; j < numHidden2; j++) 
            {
               out.writeDouble(w[HIDDEN1_INDEX][k][j]);
            }
         } // for (int k = 0; k < numHidden1; k++)

         out.writeInt(numHidden2);
         out.writeInt(numOutputs);

         for (int j=0; j < numHidden2; j++)
         {
            for (int i = 0; i < numOutputs; i++)
            {
               out.writeDouble(w[HIDDEN2_INDEX][j][i]);
            }
         } // for (int j = 0; j < numHidden2; j++)
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
      for (int m = 0; m < numInputs; m++)
      {
         for (int k = 0; k < numHidden1; k++)
         {
               w[INPUT_INEDX][m][k] = generateRandomDouble(min, max);
         }
      }

      for (int k = 0; k < numHidden1; k++)
      {
         for (int j = 0; j < numHidden2; j++)
         {
            w[HIDDEN1_INDEX][k][j] = generateRandomDouble(min, max);
         }
      }
      
      for (int j = 0; j < numHidden2; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            w[HIDDEN2_INDEX][j][i] = generateRandomDouble(min, max);
         }
      }
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
* Calculates activations for the first hidden layer (K) during training and stores pre-activation values in theta.
* Depends on a[INPUT_INEDX] holding the current input activations.
*/
   public void calculateKActivationsTraining()
   {
      double sum;
      
      int n = HIDDEN1_INDEX;
      for (int k = 0; k < numHidden1; k++)
      {
         sum = 0.0;
         for (int m = 0; m < numInputs; m++)
         {
            sum += w[n-1][m][k] * a[n-1][m];
         }
         theta[n][k] = sum;
         a[n][k] = activationFunction.apply(sum);
      } // for (int k = 0; k < numHidden1; k++)
   } // calculateKActivationsTraining()

/*
* Calculates activations for the second hidden layer (J) during training and stores pre-activation values in theta.
* Requires calculateKActivationsTraining to have populated a[HIDDEN1_INDEX].
*/
   public void calculateJActivationsTraining()
   {
      double sum;
      
      int n = HIDDEN2_INDEX;
      for (int j = 0; j < numHidden2; j++)
      {
         sum = 0.0;
         for (int k = 0; k < numHidden1; k++)
         {
            sum += w[n-1][k][j] * a[n-1][k];
         }
         theta[n][j] = sum;
         a[n][j] = activationFunction.apply(sum);
      } // for (int j = 0; j < numHidden2; j++)
   } // calculateJActivationsTraining()

/*
* Calculates the network outputs during training, populates psi for the output layer, and accumulates half-squared error.
* Returns half the squared error for the currently active training sample.
*/
   public double calculateOutputTraining()
   {
      double sum;
      double omega;
      double currentError;

      int n = OUTPUT_INDEX;
      currentError = 0.0;
      for (int i = 0; i < numOutputs; i++)
      {
         sum = 0.0;
         for (int j = 0; j < numHidden2; j++)
         {
            sum += w[n-1][j][i] * a[n-1][j];
         }
         a[n][i] = activationFunction.apply(sum);
         omega = targets[i] - a[n][i];
         psi[n][i] = omega * activationFunctionDerivative.apply(sum);
         currentError += omega * omega;
      } // for (int i = 0; i < numOutputs; i++)
      return currentError / 2.0;
   } // calculateOutputTraining()

/*
* Performs a full forward pass during training (K, J, and output layers) and returns half the squared error.
* Assumes a[INPUT_INEDX] and targets are populated for the current case.
*/
   public double runNetworkForTrain()
   {
      calculateKActivationsTraining();
      calculateJActivationsTraining();
      return calculateOutputTraining();
   }

/*
* Performs a forward pass using the current values in a[INPUT_INEDX] as input activations.
* Useful for inference or evaluation after training.
*/
   public void run()
   {
      double sum;

      int n = HIDDEN1_INDEX;
      for (int k = 0; k < numHidden1; k++)
      {
         sum = 0.0;
         for (int m = 0; m < numInputs; m++)
         {
            sum += w[n-1][m][k] * a[n-1][m];
         }
         a[n][k] = activationFunction.apply(sum);
      } // for (int k = 0; k < numHidden1; k++)
      
      n = HIDDEN2_INDEX;
      for (int j = 0; j < numHidden2; j++)
      {
         sum = 0.0;
         for (int k = 0; k < numHidden1; k++)
         {
            sum += w[n-1][k][j] * a[n-1][k];
         }
         a[n][j] = activationFunction.apply(sum);
      } // for (int j = 0; j < numHidden2; j++)

      n = OUTPUT_INDEX;
      for (int i = 0; i < numOutputs; i++)
      {
         sum = 0.0;
         for (int j = 0; j < numHidden2; j++)
         {
            sum += w[n-1][j][i] * a[HIDDEN2_INDEX][j];
         }
         a[n][i] = activationFunction.apply(sum);
      } // for (int i = 0; i < numOutputs; i++)
   } // forwardPass(double[] inputs)

/*
* Runs the network on every configured case, capturing timing metrics and copying outputs for later reporting.
*/
   public void runAllCases()
   {
      runStartTimeMillis = System.currentTimeMillis();
      for (int t = 0; t < numCases; t++)
      {
         a[INPUT_INEDX] = inputTable[t];
         run();
         System.arraycopy(a[OUTPUT_INDEX], 0, networkOutputs[t], 0, numOutputs); // Efficient array copy
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
            a[INPUT_INEDX] = inputTable[t];
            targets = groundTruths[t];

            totalError += runNetworkForTrain();

            train();
         } // for (int t = 0; t < numCases; t++)

         averageError = totalError / (double) numCases;
         epochs++;
      } // while (epochs < IterationMax && averageError > ECutoff)
      trainEndTimeMillis = System.currentTimeMillis();
   } // loopTraining()

/*
* Performs one backpropagation update using the derivatives captured in psi and theta.
* Requires runNetworkForTrain to have populated psi for the output layer before invocation.
*/
   public void train()
   {
      double omegaJ, omegaK;

      int n = HIDDEN2_INDEX;
      for (int j = 0; j < numHidden2; j++)
      {
         omegaJ = 0.0;
         for (int i = 0; i < numOutputs; i++)
         {
            omegaJ += psi[n+1][i] * w[n][j][i] ;
            w[n][j][i] += learningRate * a[n][j] * psi[n+1][i];
         }

         psi[n][j] = omegaJ * activationFunctionDerivative.apply(theta[n][j]);
      } // for (int j = 0; j < numHidden2; j++)

      n = HIDDEN1_INDEX;
      for (int k = 0; k < numHidden1; k++)
      {
         omegaK = 0.0;
         for (int j = 0; j < numHidden2; j++)
         {
            omegaK += psi[n+1][j] * w[n][k][j];
            w[n][k][j] += learningRate * a[n][k] * psi[n+1][j];
         }
         
         psi[n][k] = omegaK * activationFunctionDerivative.apply(theta[n][k]);

         for (int m = 0; m < numInputs; m++)
         {
            w[n-1][m][k] += learningRate * a[n-1][m] * psi[n][k];
         }
      } // for (int k = 0; k < numHidden1; k++)
   } // train()

/*
* Formats the first len entries of a double array as [v1, v2, ...] using formatDouble precision.
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
      System.out.println("======== Network Parameters ========");

      System.out.println("Network Architecture:");
      System.out.println(numInputs  + "-" + numHidden1 + "-" + numHidden2 + "-" + numOutputs);
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
      System.out.println("Weight Initialization Range: [" + min + ", " + max + ")");
      System.out.println("Manual Weights: " + manualWeights);
      System.out.println("Load Weights from " + inputWeightsFileName + ": " + loadWeightsFromFile);
      System.out.println("Save Weights to " + outputWeightsFileName + ": " + saveWeightsToFile);
      System.out.println("Load Truth Table from " + truthTableFileName + ": " + loadTruthTableFromCSV);
      System.out.println("Loading test cases from " + inputTableFileName);


      System.out.println("\nConfiguration Booleans:");
      System.out.println("Run Test Cases: " + runTestCases);
      System.out.println("Show Inputs: " + showInputs);
      System.out.println("Show Ground Truths: " + showGroundTruths);
   } // printNetworkParameters()

/**
 * Prints the network weights.
 */
   public void printNetworkWeights()
   {
      System.out.println("\nNetwork Weights:");

      System.out.println("Layer 1 Weights:");
      for (int m = 0; m < numInputs; m++)
      {
         for (int k = 0; k < numHidden1; k++)
         {
            System.out.print(w[INPUT_INEDX][m][k] + " ");
         }
         System.out.println();
      } // for (int m = 0; m < numInputs; m++)

      System.out.println("Layer 2 Weights:");
      for (int k = 0; k < numHidden1; k++)
      {
         for (int j = 0; j < numHidden2; j++)
         {
            System.out.print(w[HIDDEN1_INDEX][k][j] + " ");
         }
         System.out.println();
      } // for (int k = 0; k < numHidden1; k++)

      System.out.println("Layer 3 Weights:");
      for (int j = 0; j < numHidden2; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            System.out.print(w[HIDDEN2_INDEX][j][i] + " ");
         }
         System.out.println();
      } // for (int j = 0; j < numHidden2; j++)
      System.out.println();
   } // printNetworkWeights()

/**
* Main entry point for running the network.
* @param args Command-line arguments
*/
   public static void main(String[] args)
   {
      ABCDNetwork p = new ABCDNetwork();
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
} // class ABCDNetwork
