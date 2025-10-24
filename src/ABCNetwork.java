import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
* Currently, the ABCNetwork class implements a simple feedforward neural network with one hidden layer and one output.
* It supports both manual and random weight initialization, training using gradient descent,
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
public class ABCNetwork
{
   public double min, max;
   public double learningRate, ECutoff;
   public int numInputs, numHidden, numOutputs, numLayers;
   public int numCases, IterationMax;
   public double[][] w1, w2, trainingInputs, networkOutputs, groundTruths;
   public double[] inputs, targets, thetaJ, h, thetaI, f, psiI;
   public double currentError, averageError;
   public int epochs;
   public String activationName;
   public boolean training, manualWeights, runTestCases, showInputs, showGroundTruths;
   public boolean loadWeightsFromFile, saveWeightsToFile, loadTruthTableFromCSV;
   public String inputWeightsFileName, outputWeightsFileName, inputTableFileName, truthTableFileName;
   public String booleanOperation;
   public long trainTimeMillis;

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
            double exp = Math.exp(negFactor * 2 * x);
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
         
         JsonObject network = getRequiredObject(json, "network"); // Network architecture parameters
         numInputs = getRequiredInt(network, "numInputs", "network");
         numHidden = getRequiredInt(network, "numHidden", "network");
         numOutputs = getRequiredInt(network, "numOutputs", "network");
         activationName = getRequiredString(network, "activationName", "network");
         activationFunction = ACTIVATION_MAP.get(activationName);
         activationFunctionDerivative = ACTIVATION_DERIVATIVE_MAP.get(activationName);

         JsonObject trainingParams = getRequiredObject(json, "training"); // Training parameters
         learningRate = getRequiredDouble(trainingParams, "learningRate", "training");
         ECutoff = getRequiredDouble(trainingParams, "ECutoff", "training");
         IterationMax = getRequiredInt(trainingParams, "IterationMax", "training");
         numCases = getRequiredInt(trainingParams, "numCases", "training");

         JsonObject arrParams = getRequiredObject(json, "arrayParameters"); // Array initialization parameters
         min = getRequiredDouble(arrParams, "min", "arrayParameters");
         max = getRequiredDouble(arrParams, "max", "arrayParameters");
         manualWeights = getRequiredBoolean(arrParams, "manualWeights", "arrayParameters");
         loadWeightsFromFile = getRequiredBoolean(arrParams, "loadWeightsFromFile", "arrayParameters");
         saveWeightsToFile = getRequiredBoolean(arrParams, "saveWeightsToFile", "arrayParameters");
         inputWeightsFileName = getRequiredString(arrParams, "inputWeightsFileName", "arrayParameters");
         outputWeightsFileName = getRequiredString(arrParams, "outputWeightsFileName", "arrayParameters");
         loadTruthTableFromCSV = getRequiredBoolean(arrParams, "loadTruthTableFromCSV", "arrayParameters");
         inputTableFileName = getRequiredString(arrParams, "inputTableFileName", "arrayParameters");
         truthTableFileName = getRequiredString(arrParams, "truthTableFileName", "arrayParameters");

         JsonObject execution = getRequiredObject(json, "execution"); // Execution parameters
         training = getRequiredBoolean(execution, "training", "execution");
         runTestCases = getRequiredBoolean(execution, "runTestCases", "execution");
         booleanOperation = getRequiredString(execution, "booleanOperation", "execution");

         JsonObject display = getRequiredObject(json, "display"); // Display parameters
         showInputs = getRequiredBoolean(display, "showInputs", "display");
         showGroundTruths = getRequiredBoolean(display, "showGroundTruths", "display");
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

      w1 = new double[numInputs][numHidden];
      h = new double[numHidden];
      
      w2 = new double[numHidden][numOutputs];
      f = new double[numOutputs];

      if (training)
      {
         psiI = new double[numOutputs];
         thetaJ = new double[numHidden];
         thetaI = new double[numOutputs];
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
            throw new java.io.IOException("Error: CSV '" + filename + "' has invalid dimensions. Expected " + arrRows + "x" + arrCols + 
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
         int fileNumInputs = in.readInt(); // Read and check dimensions for w1
         int fileNumHidden = in.readInt();

         if (fileNumInputs != numInputs || fileNumHidden != numHidden) 
         {
            throw new java.io.IOException("Error: File dimensions for w1 (" + fileNumInputs + "-" + fileNumHidden +
                               ") do not match network (" + numInputs + "-" + numHidden + ")"); 
         }

         for (int k = 0; k < numInputs; k++) 
         {
            for (int j = 0; j < numHidden; j++) 
            {
               w1[k][j] = in.readDouble();
            }
         } // for (int k = 0; k < numInputs; k++)

         int fileNumHidden2 = in.readInt(); // Read and check dimensions for w2
         int fileNumOutputs = in.readInt();

         if (fileNumHidden2 != numHidden || fileNumOutputs != numOutputs) 
         {
            throw new IndexOutOfBoundsException("Error: File dimensions for w2 (" + fileNumHidden2 + "-" + fileNumOutputs +
                               ") do not match network (" + numHidden + "-" + numOutputs + ")");
         }

         for (int j = 0; j < numHidden; j++) 
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
         out.writeInt(numHidden);

         for (int k = 0; k < numInputs; k++) 
         {
            for (int j = 0; j < numHidden; j++) 
            {
               out.writeDouble(w1[k][j]);
            }
         } // for (int k = 0; k < numInputs; k++)

         out.writeInt(numHidden);
         out.writeInt(numOutputs);

         for (int j = 0; j < numHidden; j++) 
         {
            for (int i = 0; i < numOutputs; i++) 
            {
               out.writeDouble(w2[j][i]);
            }
         } // for (int j = 0; j < numHidden; j++)
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
      for (int j = 0; j < numHidden; j++)
      {
         for (int k = 0; k < numInputs; k++)
         {
            w1[k][j] = generateRandomDouble(min, max);
         }
      } // for (int j = 0; j < numHidden; j++)
      
      for (int i = 0; i < numOutputs; i++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            w2[j][i] = generateRandomDouble(min, max);
         }
      } // for (int i = 0; i < numOutputs; i++)
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
   public void calculateHActivationsTraining()
   {
      double sum;
      
      for (int j = 0; j < numHidden; j++)
      {
         sum = 0.0;
         for (int k = 0; k < numInputs; k++)
         {
            sum += w1[k][j] * inputs[k];
         }
         thetaJ[j] = sum;
         h[j] = activationFunction.apply(sum);
      } // for (int j = 0; j < numHidden; j++)
   } // calculateHActivationsTraining(double[] inputs)

/**
* Calculates the output of the network based on hidden layer activations.
*/
   public void calculateOutputTraining()
   {
      double sum;
      double omega;

      currentError = 0.0;
      for (int i = 0; i < numOutputs; i++)
      {
         sum = 0.0;
         for (int j = 0; j < numHidden; j++)
         {
            sum += w2[j][i] * h[j];
         }
         thetaI[i] = sum;
         f[i] = activationFunction.apply(sum);
         omega = targets[i] - f[i];
         psiI[i] = omega * activationFunctionDerivative.apply(thetaI[i]);
         currentError += omega * omega;
      } // for (int i = 0; i < numOutputs; i++)
      currentError /= 2.0;
   } // calculateOutputTraining()

/**
* Performs a forward pass through the network.
* @param inputs Input vector
*/
   public void run(double[] inputs)
   {
      double sum;

      for (int j = 0; j < numHidden; j++)
      {
         sum = 0.0;
         for (int k = 0; k < numInputs; k++)
         {
            sum += w1[k][j] * inputs[k];
         }
         h[j] = activationFunction.apply(sum);
      } // for (int j = 0; j < numHidden; j++)
      
      for (int i = 0; i < numOutputs; i++)
      {
         sum = 0.0;
         for (int j = 0; j < numHidden; j++)
         {
            sum += w2[j][i] * h[j];
         }
         f[i] = activationFunction.apply(sum);
      } // for (int i = 0; i < numOutputs; i++)
   } // forwardPass(double[] inputs)

/**
* Runs the network on all training inputs and stores the outputs.
*/
   public void runAllCases()
   {
      for (int t = 0; t < numCases; t++)
      {
         run(trainingInputs[t]);
         System.arraycopy(f, 0, networkOutputs[t], 0, numOutputs); // Efficient array copy
      }
   } // runAllCases()

/**
* Trains the network using the provided training data.
*/
   public void loopTraining()
   {
      long utcTimeInMillis = System.currentTimeMillis();

      epochs = 0;
      averageError = Double.MAX_VALUE;
      while (epochs < IterationMax && averageError > ECutoff)
      {
         averageError = 0.0;
         for (int t = 0; t < numCases; t++)
         {
            inputs = trainingInputs[t];
            targets = groundTruths[t];
            
            calculateHActivationsTraining();
            calculateOutputTraining();

            averageError += currentError;
            train();
         } // for (int t = 0; t < trainingInputs.length; t++)

         averageError /= (double) numCases;
         epochs++;
      } // while (epochs < IterationMax && averageError > ECutoff)
      trainTimeMillis = System.currentTimeMillis() - utcTimeInMillis;
   } // loopTraining()

/**
* Performs a single optimization step to update weights using the backpropagation algorithm.
*/
   public void train()
   {
      double omega, psiJ;

      for (int j = 0; j < numHidden; j++)
      {
         omega = 0.0;
         for (int i = 0; i < numOutputs; i++)
         {
            omega += psiI[i] * w2[j][i];
            w2[j][i] += learningRate * h[j] * psiI[i];
         }

         psiJ = omega * activationFunctionDerivative.apply(thetaJ[j]);

         for (int k = 0; k < numInputs; k++)
         {
            w1[k][j] += learningRate * inputs[k] * psiJ;
         }
      } // for (int j = 0; j < numHidden; j++)
   } // train()

/**
* Prints the results of the training process, including convergence status and final error.
*/
   public void printTrainingResults()
   {
      System.out.println("Training Results:");

      System.out.println("Training Time: " + trainTimeMillis + " milliseconds");
      if (epochs == IterationMax)
         System.out.println("Warning: Training did not converge to desired error value within " +
                            IterationMax + " iterations. Final error: " + averageError);
      
      else if (averageError <= ECutoff)
         System.out.println("Training converged successfully after " + epochs + " iterations. Final error: " + averageError); 
   } // printTrainingResults()

/**
 * Prints the results of the network run.
* @param includeInputs Whether to include input values
* @param includeGroundTruths Whether to include ground truth values
*/
   public void printRunResults()
   {
      System.out.println("Run Results:");
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
      System.out.println(numInputs  + "-" + numHidden + "-" + numOutputs);
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
      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            System.out.print(w1[k][j] + " ");
         }
         System.out.println();
      } // for (int k = 0; k < numInputs; k++)

      System.out.println("Layer 2 Weights:");
      for (int i = 0; i < numHidden; i++)
      {
         for (int j = 0; j < numOutputs; j++)
         {
            System.out.print(w2[i][j] + " ");
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
      ABCNetwork p = new ABCNetwork();
      String configFileName = "network-config.json";

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
