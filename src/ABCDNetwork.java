import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
* Currently, the ABCNetwork class implements a simple feedforward neural network with one hidden layer and one output.
* It supports both manual and random weight initialization, training using backpropagation,
* and evaluation on input data. The network can be configured for different activation functions and learning rates.
* In order to compile this program, you need to have the gson library installed and add it to the classpath.
*
* To compile this program run the following command in the root directory of the project:
* javac -cp ".;lib\gson-2.10.1.jar" ABCNetwork.java
*
* To run this program run the following command in the root directory of the project:
* java -cp ".;lib\gson-2.10.1.jar" ABCNetwork
*
* External Dependencies:
* This class uses the Gson library (version 2.10.1) by Google for JSON parsing and configuration management.
* Gson documentation: https://github.com/google/gson
*
* Gson classes and methods used:
* - com.google.gson.Gson: Main class for JSON serialization/deserialization
*   - fromJson(Reader, Class): Parses JSON from a file reader into Java objects
* - com.google.gson.JsonObject: Represents a JSON object as a tree of JsonElements
*   - get(): Retrieves a member by name
*   - getAsJsonObject(): Gets a nested JSON object
*   - getAsInt(): Converts element to int
*   - getAsDouble(): Converts element to double
*   - getAsString(): Converts element to String
*   - getAsBoolean(): Converts element to boolean
*
* @author Vouk Praun-Petrovic
* @version September 9, 2024
*/
public class ABCDNetwork
{
   public double min, max;
   public double learningRate, ECutoff;
   public int numHiddenLayers, numInputs, numHidden1, numHidden2, numOutputs;
   public int numCases, IterationMax, epochs;
   public double[][] w1, w2, w3, h, trainingInputs, networkOutputs, groundTruths, psi;
   public double[] inputs, targets, thetaK, thetaJ, f;
   public double averageError;
   public String activationName;
   public boolean training, manualWeights, runTestCases, showInputs, showGroundTruths;
   public boolean loadWeightsFromFile, saveWeightsToFile, loadTruthTableFromCSV;
   public String inputWeightsFileName, outputWeightsFileName, inputTableFileName, truthTableFileName;
   public String booleanOperation;
   public long trainStartTimeMillis, runStartTimeMillis, trainEndTimeMillis, runEndTimeMillis;
   public static final int K_ACTIVATIONS_INDEX = 1;
   public static final int J_ACTIVATIONS_INDEX = 2;
   public static final int OUTPUT_ACTIVATIONS_INDEX = 3;

/**
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

/**
* Initializes network parameters by loading configuration from the network-config.json file.
* Reads network architecture, training parameters, weight settings, execution options, and display settings.
* @throws java.io.IOException If config file is missing or contains invalid/missing settings
*/
   public void initializeNetworkParams(String configFileName) throws java.io.IOException
   {
      try 
      {
         JsonObject json = new Gson().fromJson(new java.io.FileReader(configFileName), JsonObject.class);
         
         String networkSectionHeader = "network";
         JsonObject network = getRequiredObject(json, networkSectionHeader); // Network architecture parameters
         numHiddenLayers = 2; // Will later be read in from config file in this section when N-layer is implemented
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

/**
* Gets a required JSON object from a parent JSON object.
* @param parent The parent JSON object
* @param key The key of the required JSON object
* @return The required JSON object
* @throws java.io.IOException If the required JSON object is not found
*/
   private static JsonObject getRequiredObject(JsonObject parent, String key) throws java.io.IOException
   {
      if (parent.get(key) == null) 
      {
         throw new java.io.IOException("Missing required section '" + key + "' in config file");
      }
      return parent.getAsJsonObject(key);
   } // getRequiredObject(JsonObject parent, String key)

/**
* Gets a required integer from a JSON object.
* @param obj The JSON object
* @param key The key of the required integer
* @param section The section of the JSON object
* @return The required integer
* @throws java.io.IOException If the required integer is not found
*/
   private static int getRequiredInt(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      return obj.get(key).getAsInt();
   } // getRequiredInt(JsonObject obj, String key, String section)

/**
* Gets a required double from a JSON object.
* @param obj The JSON object
* @param key The key of the required double
* @param section The section of the JSON object
* @return The required double
* @throws java.io.IOException If the required double is not found
*/
   private static double getRequiredDouble(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      return obj.get(key).getAsDouble();
   } // getRequiredDouble(JsonObject obj, String key, String section)

/**
* Gets a required string from a JSON object.
* @param obj The JSON object
* @param key The key of the required string
* @param section The section of the JSON object
* @return The required string
* @throws java.io.IOException If the required string is not found
*/
   private static String getRequiredString(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      return obj.get(key).getAsString();
   } // getRequiredString(JsonObject obj, String key, String section)

/**
* Gets a required boolean from a JSON object.
* @param obj The JSON object
* @param key The key of the required boolean
* @param section The section of the JSON object
* @return The required boolean
* @throws java.io.IOException If the required boolean is not found
*/
   private static boolean getRequiredBoolean(JsonObject obj, String key, String section) throws java.io.IOException
   {
      if (obj.get(key) == null) 
      {
         throw new java.io.IOException("Missing required setting '" + key + "' in section '" + section + "'");
      }
      return obj.get(key).getAsBoolean();
   } // getRequiredBoolean(JsonObject obj, String key, String section)

/**
* Allocates memory for the network's arrays based on configuration.
*/
   public void allocateNetworkArrays()
   {
      trainingInputs = new double[numCases][numInputs];
      networkOutputs = new double[numCases][numOutputs];

      w1 = new double[numInputs][numHidden1];
      w2 = new double[numHidden1][numHidden2];
      w3 = new double[numHidden2][numOutputs];

      h = new double[numHiddenLayers + 1][Math.max(numHidden1, numHidden2)]; // one layer added so indexing can begin from 1
      f = new double[numOutputs];

      if (training)
      {
         psi = new double[numHiddenLayers + 2][Math.max(numHidden1, Math.max(numHidden2, numOutputs))];  // one psi array for each layer, so its numHiddenLayers + 2
         thetaK = new double[numHidden1];
         thetaJ = new double[numHidden2];
      }

      if (training | showGroundTruths)
      {
         groundTruths = new double[numCases][numOutputs];
      }
   } // allocateNetworkArrays()

/**
* Populates the network's weight arrays either with manual values or random values.
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

/**
* Sets predefined weights for the network.
*/
   public void setPredefinedWeights()
   {
      // Correct order: w1[input][hidden]
      w1[0][0] = 0.9404045278126735;
      w1[0][1] = -1.0121547995672862;
      w1[1][0] = 0.2241493210468354;
      w1[1][1] = -1.432402348525032;

      w2[0][0] = 0.251086609938219;
      w2[1][0] = -0.41775164313179797;
   } // setPredefinedWeights()


/**
* Populates the input values and ground truth values for the training set.
*/
   public void manuallySetInputsAndTruthTable()
   {
      trainingInputs[0][0] = 0.0;
      trainingInputs[0][1] = 0.0;
      trainingInputs[1][0] = 1.0;
      trainingInputs[1][1] = 0.0;
      trainingInputs[2][0] = 0.0;
      trainingInputs[2][1] = 1.0;
      trainingInputs[3][0] = 1.0;
      trainingInputs[3][1] = 1.0;

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

/**
* Loads the input and truth table from a CSV file.
* @param trainingInputs The input data
* @param groundTruths The truth data
* @throws java.io.IOException If the input or truth table is not found
*/
   public void loadInputsAndTruthTable() throws java.io.IOException
   {
      loadDoubleArrfromCSV(trainingInputs, numCases, numInputs, inputTableFileName); // Load the input data
      loadDoubleArrfromCSV(groundTruths, numCases, numOutputs, truthTableFileName); // Load the truth data
   } // loadInputsAndTruthTable()

/**
* Loads a double array from a CSV file.
* @param arr The array to load
* @param arrRows The number of rows in the array
* @param arrCols The number of columns in the array
* @param filename The name of the CSV file
* @throws java.io.IOException If the array is not found
*/
   public static void loadDoubleArrfromCSV(double[][] arr, int arrRows, int arrCols, String filename) throws java.io.IOException
   {
      try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(filename)))
      {
         String line = reader.readLine(); // Requires first line to be the dimensions of the rest of theCSV
         int numRows = Integer.parseInt(line.split(",")[0]);
         int numCols = Integer.parseInt(line.split(",")[1]);
         String[] values;

         if (numRows != arrRows || numCols != arrCols)
         {
            throw new java.io.IOException("Error:xCSV '" + filename + "' has invalid dimensions. Expected " + arrRows + "x" + arrCols + 
               " but found " + numRows + "x" + numCols + ".");
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


/**
* Loads weights from a file.
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
            throw new java.io.IOException("Error: File dimensions for w1 (" + fileNumInputs + "-" + fileNumHidden1 +
                               ") do not match network (" + numInputs + "-" + numHidden1 + ")"); 
         }

         for (int m = 0; m < numInputs; m++) 
         {
            for (int k = 0; k < numHidden1; k++) 
            {
               w1[m][k] = in.readDouble();
            }
         } // for (int m = 0; m < numInputs; m++)

         fileNumHidden1 = in.readInt();
         int fileNumHidden2 = in.readInt();

         if (fileNumHidden1 != numHidden1 || fileNumHidden2 != numHidden2) 
         {
            throw new java.io.IOException("Error: File dimensions for w2 (" + fileNumInputs + "-" + fileNumHidden1 +
                               ") do not match network (" + numInputs + "-" + numHidden1 + ")"); 
         }

         for (int k = 0; k < numHidden1; k++) 
         {
            for (int j = 0; j < numHidden2; j++) 
            {
               w1[k][j] = in.readDouble();
            }
         } // for (int k = 0; k < numInputs; k++)

         fileNumHidden2 = in.readInt(); // Read and check dimensions for w2
         int fileNumOutputs = in.readInt();

         if (fileNumHidden2 != numHidden2 || fileNumOutputs != numOutputs) 
         {
            throw new IndexOutOfBoundsException("Error: File dimensions for w3 (" + fileNumHidden2 + "-" + fileNumOutputs +
                               ") do not match network (" + numHidden2 + "-" + numOutputs + ")");
         }

         for (int j = 0; j < numHidden2; j++) 
         {
            for (int i = 0; i < numOutputs; i++) 
            {
               w2[j][i] = in.readDouble();
            }
         } // for (int j = 0; j < numHidden; j++)
      } // try()
      catch (java.io.IOException e) 
      {
         throw new java.io.IOException("Error opening file for loading weights: " + e.getMessage(), e);
      }
   } // loadWeightsFromFile()

/**
* Saves current model weights to a binary file.
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
               out.writeDouble(w1[m][k]);
            }
         } // for (int k = 0; k < numInputs; k++)

         out.writeInt(numHidden1);
         out.writeInt(numHidden2);

         for (int k = 0; k < numHidden1; k++) 
         {
            for (int j = 0; j < numHidden2; j++) 
            {
               out.writeDouble(w2[k][j]);
            }
         } // for (int j = 0; j < numHidden; j++)

         out.writeInt(numHidden2);
         out.writeInt(numOutputs);

         for (int j=0; j < numHidden2; j++)
         {
            for (int i = 0; i < numOutputs; i++)
            {
               out.writeDouble(w3[j][i]);
            }
         } // for (int j = 0; j < numHidden2; j++)
      } // try()
      catch (java.io.IOException e) 
      {
         throw new java.io.IOException("Error opening file for saving weights: " + e.getMessage(), e);
      }
   } // saveWeights()

/**
* Generates random weights for the network within the specified range.
*/
   public void generateRandomWeights()
   {
      for (int m = 0; m < numInputs; m++)
      {
         for (int k = 0; k < numHidden1; k++)
         {
               w1[m][k] = generateRandomDouble(min, max);
         }
      } // for (int j = 0; j < numHidden; j++)

      for (int k = 0; k < numHidden1; k++)
      {
         for (int j = 0; j < numHidden2; j++)
         {
            w2[k][j] = generateRandomDouble(min, max);
         }
      }
      
      for (int j = 0; j < numHidden2; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            w3[j][i] = generateRandomDouble(min, max);
         }
      } // for (int j = 0; j < numHidden2; j++)
   } // generateRandomWeights()

/**
 * Generates a random double within the specified range.
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @return Random double between min and max
 */
   public static double generateRandomDouble(double min, double max)
   {
      return Math.random() * (max - min) + min;
   }

/**
* Calculates the activations for the hidden layer neurons.
* @param inputs The input vector to the network.
*/
   public void calculateKActivationsTraining()
   {
      double sum;
      
      for (int k = 0; k < numHidden1; k++)
      {
         sum = 0.0;
         for (int m = 0; m < numInputs; m++)
         {
            sum += w1[m][k] * inputs[m];
         }
         thetaK[k] = sum;
         h[K_ACTIVATIONS_INDEX][k] = activationFunction.apply(sum);
      } // for (int j = 0; j < numHidden; j++)
   } // calculateHActivationsTraining(double[] inputs)

/**
 * Calculates the activations for the hidden layer neurons.
 */
   public void calculateJActivationsTraining()
   {
      double sum;
      
      for (int j = 0; j < numHidden2; j++)
      {
         sum = 0.0;
         for (int k = 0; k < numHidden1; k++)
         {
            sum += w2[k][j] * h[K_ACTIVATIONS_INDEX][k];
         }
         thetaJ[j] = sum;
         h[J_ACTIVATIONS_INDEX][j] = activationFunction.apply(sum);
      }
   } // calculateJActivationsTraining()

/**
* Calculates the output of the network based on hidden layer activations.
* @return The error of the network.
*/
   public double calculateOutputTraining()
   {
      double sum;
      double omega;
      double currentError;

     currentError = 0.0;
      for (int i = 0; i < numOutputs; i++)
      {
         sum = 0.0;
         for (int j = 0; j < numHidden2; j++)
         {
            sum += w3[j][i] * h[J_ACTIVATIONS_INDEX][j];
         }
         f[i] = activationFunction.apply(sum);
         omega = targets[i] - f[i];
         psi[OUTPUT_ACTIVATIONS_INDEX][i] = omega * activationFunctionDerivative.apply(sum);
         currentError += omega * omega;
      } // for (int i = 0; i < numOutputs; i++)
      return currentError / 2.0;
   } // calculateOutputTraining()

/**
 * Runs the network for training and returns the error.
 * @return The error of the network.
 */
   public double runNetworkForTrain()
   {
      calculateKActivationsTraining();
      calculateJActivationsTraining();
      return calculateOutputTraining();
   }

/**
* Performs a forward pass through the network.
* @param inputs Input vector
*/
   public void run(double[] inputs)
   {
      double sum;

      for (int k = 0; k < numHidden1; k++)
      {
         sum = 0.0;
         for (int m = 0; m < numInputs; m++)
         {
            sum += w1[m][k] * inputs[m];
         }
         h[K_ACTIVATIONS_INDEX][k] = activationFunction.apply(sum);
      } // for (int j = 0; j < numHidden; j++)
      

      for (int j = 0; j < numHidden2; j++)
      {
         sum = 0.0;
         for (int k = 0; k < numHidden1; k++)
         {
            sum += w2[k][j] * h[K_ACTIVATIONS_INDEX][k];
         }
         h[J_ACTIVATIONS_INDEX][j] = activationFunction.apply(sum);
      }

      for (int i = 0; i < numOutputs; i++)
      {
         sum = 0.0;
         for (int j = 0; j < numHidden2; j++)
         {
            sum += w3[j][i] * h[J_ACTIVATIONS_INDEX][j];
         }
         f[i] = activationFunction.apply(sum);
      } // for (int i = 0; i < numOutputs; i++)
   } // forwardPass(double[] inputs)

/**
* Runs the network on all training inputs and stores the outputs.
*/
   public void runAllCases()
   {
      runStartTimeMillis = System.currentTimeMillis();
      for (int t = 0; t < numCases; t++)
      {
         run(trainingInputs[t]);
         System.arraycopy(f, 0, networkOutputs[t], 0, numOutputs); // Efficient array copy
      }
      runEndTimeMillis = System.currentTimeMillis();
   } // runAllCases()

/**
* Trains the network using the provided training data.
*/
   public void loopTraining()
   {
      epochs = 0;
      trainStartTimeMillis = System.currentTimeMillis();
      averageError = Double.MAX_VALUE;
      while (epochs < IterationMax && averageError > ECutoff)
      {
         averageError = 0.0;
         for (int t = 0; t < numCases; t++)
         {
            inputs = trainingInputs[t];
            targets = groundTruths[t];

            averageError += runNetworkForTrain();

            train();
         } // for (int t = 0; t < trainingInputs.length; t++)

         averageError /= (double) numCases;
         epochs++;
      } // while (epochs < IterationMax && averageError > ECutoff)
      trainEndTimeMillis = System.currentTimeMillis();
   } // loopTraining()

/**
* Performs a single optimization step to update weights using the backpropagation algorithm.
*/
   public void train()
   {
      double omegaJ, omegaK;

      for (int j = 0; j < numHidden2; j++)
      {
         omegaJ = 0.0;
         for (int i = 0; i < numOutputs; i++)
         {
            omegaJ += psi[OUTPUT_ACTIVATIONS_INDEX][i] * w3[j][i] ;
            w3[j][i] += learningRate * h[J_ACTIVATIONS_INDEX][j] * psi[OUTPUT_ACTIVATIONS_INDEX][i];
         }

         psi[J_ACTIVATIONS_INDEX][j] = omegaJ * activationFunctionDerivative.apply(thetaJ[j]);
      }

      for (int k = 0; k < numHidden1; k++)
      {
         omegaK = 0.0;
         for (int j = 0; j < numHidden2; j++)
         {
            omegaK += psi[J_ACTIVATIONS_INDEX][j] * w2[k][j];
            w2[k][j] += learningRate * h[K_ACTIVATIONS_INDEX][k] * psi[J_ACTIVATIONS_INDEX][j];
         }
      
         psi[K_ACTIVATIONS_INDEX][k] = omegaK * activationFunctionDerivative.apply(thetaK[k]);

         for (int m = 0; m < numInputs; m++)
         {
            w1[m][k] += learningRate * inputs[m] * psi[K_ACTIVATIONS_INDEX][k];
         }
      }
   } // train()

/**
* Prints the results of the training process, including convergence status and final error.
*/
   public void printTrainingResults()
   {
      System.out.println("Training Results:");

      System.out.println("Training Time: " + (trainEndTimeMillis - trainStartTimeMillis) + " milliseconds");
      if (epochs == IterationMax)
         System.out.println("Warning: Training did not converge to desired error value within " +
                            IterationMax + " iterations. Final error: " + averageError);
      
      else if (averageError <= ECutoff)
         System.out.println("Training converged successfully after " + epochs + " iterations. Final error: " + averageError + "\n"); 
   } // printTrainingResults()

/**
 * Prints the results of the network run.
* @param includeInputs Whether to include input values
* @param includeGroundTruths Whether to include ground truth values
*/
   public void printRunResults()
   {
      System.out.println("Run Results:");
      System.out.println("Run Time: " + (runEndTimeMillis - runStartTimeMillis) + " milliseconds");
      for (int t = 0; t < numCases; t++)
      {
         if (showInputs)
            System.out.print("Inputs: " + java.util.Arrays.toString(trainingInputs[t]) + " ");

         if (showGroundTruths)
            System.out.print("Ground Truth: " + java.util.Arrays.toString(groundTruths[t]) + " ");

         System.out.println("Output: " + java.util.Arrays.toString(networkOutputs[t]));
      } // for (int t = 0; t < numCases; t++)
   } // printRunResults()

/**
 * Prints the network parameters.
* @param training Whether to include training-specific parameters
*/
   public void printNetworkParameters()
   {
      System.out.println("======== Network Parameters ========");

      System.out.println("Network Architecture:");
      System.out.println(numInputs  + "-" + numHidden1 + "-" + numHidden2 + "-" + numOutputs);
      System.out.println("Learning Rate: " + learningRate);
      System.out.println("Activation Function: " + activationName);

      if (training) 
      {
         System.out.println("\nTraining Configuration:");
         System.out.println("Number of Training Cases: " + numCases);
         System.out.println("Training Error Cutoff: " + ECutoff);
         System.out.println("Max Training Iterations: " + IterationMax);
      } // if (training)
      
      System.out.println("\nWeights Initialization:");
      System.out.println("Weight Initialization Range: [" + min + ", " + max + "]");
      System.out.println("Manual Weights: " + manualWeights);
      System.out.println("Load Weights from " + inputWeightsFileName + ": " + loadWeightsFromFile);
      System.out.println("Save Weights to " + outputWeightsFileName + ": " + saveWeightsToFile);

      System.out.println("\nConfiguration Booleans:");
      System.out.println("Run Test Cases: " + runTestCases);
      System.out.println("Show Inputs: " + showInputs);
      System.out.println("Show Ground Truths: " + showGroundTruths);

      System.out.println("Display Configuration:");
      System.out.println("Show Inputs: " + showInputs);
      System.out.println("Show Ground Truths: " + showGroundTruths + "\n");
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
            System.out.print(w1[m][k] + " ");
         }
         System.out.println();
      } // for (int m = 0; m < numInputs; m++)

      System.out.println("Layer 2 Weights:");
      for (int k = 0; k < numHidden1; k++)
      {
         for (int j = 0; j < numHidden2; j++)
         {
            System.out.print(w1[k][j] + " ");
         }
         System.out.println();
      } // for (int k = 0; k < numInputs; k++)

      System.out.println("Layer 3 Weights:");
      for (int j = 0; j < numHidden2; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            System.out.print(w2[j][i] + " ");
         }
         System.out.println();
      } // for (int i = 0; i < numHidden; i++)
      System.out.println();
   } // printNetworkWeights()

/**
* Main entry point for running the network.
* @param args Command-line arguments
*/
   public static void main(String[] args)
   {
      ABCDNetwork p = new ABCDNetwork();
      String configFileName = "network-config.json";

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
} // class ABCNetwork
