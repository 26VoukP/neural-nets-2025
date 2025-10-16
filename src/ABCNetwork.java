/**
* Currently, the ABCNetwork class implements a simple feedforward neural network with one hidden layer and one output.
* It supports both manual and random weight initialization, training using gradient descent,
* and evaluation on input data. The network can be configured for different activation functions and learning rates.
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
   public double[][] w1, layer1Deltas, w2, layer2Deltas, trainingInputs, networkOutputs, groundTruths;
   public double[] thetaJ, h, thetaI, f, psiI;
   public double averageError;
   public int epochs;
   public String activationName;
   public boolean training, manualWeights, runTestCases, showInputs, showGroundTruths, loadWeightsFromFile, saveWeightsToFile;
   public String inputWeightsFileName, outputWeightsFileName;
   public String booleanOperation;

/**
* Functional interface for activation functions and their derivatives.
*/
   @FunctionalInterface
   public interface Function
   {
      double apply(double x);
   }

   private static final java.util.HashMap<String, Function> ACTIVATION_MAP = new java.util.HashMap<>();
   static
   {
      ACTIVATION_MAP.put("sigmoid", x -> 1.0 / (1.0 + Math.exp(-x)));
      ACTIVATION_MAP.put("linear", x -> x);
   }
   private static final java.util.HashMap<String, Function> ACTIVATION_DERIVATIVE_MAP = new java.util.HashMap<>();
   static
   {
      ACTIVATION_DERIVATIVE_MAP.put("sigmoid", x -> {
         double fx = ACTIVATION_MAP.get("sigmoid").apply(x);
         return fx * (1.0 - fx);
      });
      ACTIVATION_DERIVATIVE_MAP.put("linear", x -> 1.0);
   }

   public Function activationFunction, activationFunctionDerivative;

/**
* Runs the network with hardcoded parameters for demonstration or testing.
*/
   public void initializeNetworkParams()
   {
      numInputs = 2;
      numHidden = 5;
      numOutputs = 3;
      learningRate = 0.3;
      numCases = 4;
      IterationMax = 100000;
      ECutoff = 0.0002;
      min = 0.1;
      max = 1.5;

      activationName = "sigmoid";
      activationFunction = ACTIVATION_MAP.get(activationName);
      activationFunctionDerivative = ACTIVATION_DERIVATIVE_MAP.get(activationName);

      inputWeightsFileName = "nothing.bin";
      outputWeightsFileName = "outputWeights.bin";

      booleanOperation = "CUSTOM"; // Options: "OR", "AND", "XOR", "CUSTOM"

      training = true;
      runTestCases = true;
      showInputs = true;
      showGroundTruths = true;
      manualWeights = false; // Only valid for a 2-2-1 network
      loadWeightsFromFile = false;
      saveWeightsToFile = false;
   } // initializeNetworkParams()

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
         layer1Deltas = new double[numInputs][numHidden];
         layer2Deltas = new double[numHidden][numOutputs];
      }

      if (training | showGroundTruths)
      {
         groundTruths = new double[numCases][numOutputs];
      }
   } // allocateNetworkArrays(boolean training)

/**
* Populates the network's weight arrays either with manual values or random values.
*/
   public void populateNetworkArrays() throws java.io.IOException
   {
      populateInputsAndTruthTable();

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

   } // populateNetworkArrays(boolean manualWeights)

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
            throw new IndexOutOfBoundsException("Warning: File dimensions for w1 (" + fileNumInputs + "-" + fileNumHidden +
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
   }

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

         out.writeInt(numInputs);    // saves dimensions of network to the file
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
      } // for (int i = 0; i < numOutputs; i++)W
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
* Calculates the mean squared error between target and output.
* @param T Target value
* @param F Output value
* @return Error value
*/
   public double calculateError(double[] targets)
   {
      double doubleError = 0.0;

      for (int i = 0; i < numOutputs; i++)
      {
         double diff = targets[i] - f[i];
         doubleError += (diff * diff);
      }
      return doubleError / 2.0;
   } // calculateError()

/**
* Calculates the activations for the hidden layer neurons.
* @param inputs The input vector to the network.
*/
   public void calculateHActivationsTraining(double[] inputs)
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

      for (int i = 0; i < numOutputs; i++)
      {
         sum = 0.0;
         for (int j = 0; j < numHidden; j++)
         {
            sum += w2[j][i] * h[j];
         }
         thetaI[i] = sum;
         f[i] = activationFunction.apply(sum);
      } // for (int i = 0; i < numOutputs; i++)
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
* @param trainingInputs Input data for training
* @param trainingOutputs Target outputs for training
*/
   public void loopTraining()
   {
      epochs = 0;
      averageError = Double.MAX_VALUE;
      while (epochs < IterationMax && averageError > ECutoff)
      {
         averageError = 0.0;
         for (int t = 0; t < numCases; t++)
         {
            double[] inputs = trainingInputs[t];
            double[] targets = groundTruths[t];
            
            calculateHActivationsTraining(inputs);
            calculateOutputTraining();

            averageError += calculateError(targets);
            train(inputs, targets);
         } // for (int t = 0; t < trainingInputs.length; t++)

         averageError /= (double) numCases;
         epochs++;
      } // while (epochs < IterationMax && averageError > ECutoff)
   } // loopTraining()

/**
* Performs a single optimization step to update weights.
* @param inputs Input vector
* @param hiddenLayer Activations of the hidden layer
* @param output Output of the network
* @param T Target value
*/
   public void train(double[] inputs, double[] T)
   {
      double omega, psiJ, derivE_Wkj;

      for (int i = 0; i < numOutputs; i++)
      {
         psiI[i] = (T[i] - f[i]) * activationFunctionDerivative.apply(thetaI[i]);
         for (int j = 0; j < numHidden; j++)
         {
            layer2Deltas[j][i] = learningRate * psiI[i] * h[j];
         }
      } // for (int i = 0; i < numOutputs; i++)

      for (int j = 0; j < numHidden; j++)
      {
         omega = 0.0;
         for (int i = 0; i < numOutputs; i++)
         {
            omega += psiI[i] * w2[j][i];
         }

         psiJ = omega * activationFunctionDerivative.apply(thetaJ[j]);

         for (int k = 0; k < numInputs; k++)
         {
            derivE_Wkj = -psiJ * inputs[k];
            layer1Deltas[k][j] = -learningRate * derivE_Wkj;
         } // for (int k = 0; k < numInputs; k++)
      } // for (int j = 0; j < numHidden; j++)
      applyWeightDeltas();
   } // train(double[] inputs, double T)

/**
* Applies the calculated weight deltas to update the network's weights.
*/
   public void applyWeightDeltas()
   {
      for (int j = 0; j < numHidden; j++)
      {
         for (int k = 0; k < numInputs; k++)
         {
            w1[k][j] += layer1Deltas[k][j];
         }
      } // for (int j = 0; j < numHidden; j++)

      for (int i = 0; i < numOutputs; i++)
      {
         for (int j = 0; j < numHidden; j++)
            w2[j][i] += layer2Deltas[j][i];
      } // for (int i = 0; i < numOutputs; i++)
   } // applyWeightDeltas()

/**
* Prints the results of the training process, including convergence status and final error.
*/
   public void printTrainingResults()
   {
      System.out.println("Training Results:");

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
      System.out.println("Network Parameters:");

      System.out.println(numInputs  + "-" + numHidden + "-" + numOutputs);
      System.out.println("Number of Hidden Neurons: " + numHidden);
      System.out.println("Number of Outputs: " + numOutputs);
      System.out.println("Learning Rate: " + learningRate);
      System.out.println("Weight Initialization Range: [" + min + ", " + max + "]");
      System.out.println("Activation Function: " + activationName);
      System.out.println("Network Output: " + booleanOperation + "\n");

      if (training) 
      {
         System.out.println("Number of Training Cases: " + numCases);
         System.out.println("Training Error Cutoff: " + ECutoff);
         System.out.println("Max Training Iterations: " + IterationMax);
      }
      
      if (loadWeightsFromFile)
      {
         System.out.println("Loading Weights from: " + inputWeightsFileName);
      }
      if (saveWeightsToFile)
      {
         System.out.println("Saving Weights to: " + outputWeightsFileName);
      }

      System.out.println("\nConfiguration Booleans:");
      System.out.println("Training Mode: " + training);
      System.out.println("Manual Weights: " + manualWeights);
      System.out.println("Run Test Cases: " + runTestCases);
      System.out.println("Show Inputs: " + showInputs);
      System.out.println("Show Ground Truths: " + showGroundTruths);
      System.out.println("Load Weights from File: " + loadWeightsFromFile);
      System.out.println("Save Weights to File: " + saveWeightsToFile + "\n");
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
* Populates the input values and ground truth values for the training set.
*/
   @SuppressWarnings("ConvertToStringSwitch")
   public void populateInputsAndTruthTable()
   {
      if (!booleanOperation.equals("CUSTOM"))
      {
         trainingInputs[0][0] = 0.0;
         trainingInputs[0][1] = 0.0;

         trainingInputs[1][0] = 1.0;
         trainingInputs[1][1] = 0.0;

         trainingInputs[2][0] = 0.0;
         trainingInputs[2][1] = 1.0;

         trainingInputs[3][0] = 1.0;
         trainingInputs[3][1] = 1.0;

         if (training || showGroundTruths)
         {
            if (booleanOperation.equals("OR"))
            {
               groundTruths[0][0] = 0.0;
               groundTruths[1][0] = 1.0;
               groundTruths[2][0] = 1.0;
               groundTruths[3][0] = 1.0;
            }
            else if (booleanOperation.equals("AND"))
            {
               groundTruths[0][0] = 0.0;
               groundTruths[1][0] = 0.0;
               groundTruths[2][0] = 0.0;
               groundTruths[3][0] = 1.0;
            }
            else if (booleanOperation.equals("XOR"))
            {
               groundTruths[0][0] = 0.0;
               groundTruths[1][0] = 1.0;
               groundTruths[2][0] = 1.0;
               groundTruths[3][0] = 0.0;
            }
            else 
            {
               throw new IllegalArgumentException("Error populating truth table for '" + booleanOperation + "' operation.");
            }
         } // if (training || showGroundTruths)
      } // if (!booleanOperation.equals("CUSTOM"))
      else
      {
         trainingInputs[0][0] = 0.0;
         trainingInputs[0][1] = 0.0;

         trainingInputs[1][0] = 0.0;
         trainingInputs[1][1] = 1.0;

         trainingInputs[2][0] = 1.0;
         trainingInputs[2][1] = 0.0;

         trainingInputs[3][0] = 1.0;
         trainingInputs[3][1] = 1.0;

         if (training || showGroundTruths)
         {
            groundTruths[0][0] = 0.0;  groundTruths[0][1] = 0.0;  groundTruths[0][2] = 0.0;
            groundTruths[1][0] = 0.0;  groundTruths[1][1] = 1.0;  groundTruths[1][2] = 1.0;
            groundTruths[2][0] = 0.0;  groundTruths[2][1] = 1.0;  groundTruths[2][2] = 1.0;
            groundTruths[3][0] = 1.0;  groundTruths[3][1] = 1.0;  groundTruths[3][2] = 0.0;
         }
      } // else
   } // populateInputsAndTruthTable()

/**
* Main entry point for running the network.
* @param args Command-line arguments
*/
   public static void main(String[] args)
   {
      ABCNetwork p = new ABCNetwork();

      p.initializeNetworkParams();
      p.printNetworkParameters();
      p.allocateNetworkArrays();

      try
      {
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
      catch (java.io.IOException | IllegalArgumentException | IndexOutOfBoundsException e)
      {
         System.err.println(e.getMessage());
      }
   } // main(String[] args)
} // class ABCNetwork
