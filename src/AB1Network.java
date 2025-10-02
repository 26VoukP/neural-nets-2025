/**
* Currently, the AB1Network class implements a simple feedforward neural network with one hidden layer and one output.
* It supports both manual and random weight initialization, training using gradient descent,
* and evaluation on input data. The network can be configured for different activation functions and learning rates.
*
* @author Vouk Praun-Petrovic
* @version September 9, 2024
*/
public class AB1Network
{
   public double min, max;
   public double learningRate, ECutoff;
   public int numInputs, numHidden, numOutputs, numLayers;
   public int numCases, IterationMax;
   public double[][] w1, layer1Deltas;
   public double[][] trainingInputs;
   public double[] w2, layer2Deltas, thetaJ, h, groundTruths, networkOutputs;
   public double thetaI, output, averageError;
   public int epochs;
   public String activationName;
   public boolean training, manualWeights, runTestCases, showInputs, showGroundTruths;

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
* Allocates memory for the network's arrays based on configuration.
* @param training Whether the network is in training mode
* @param showGroundTruths Whether to display ground truths during training
*/
   public void allocateNetworkArrays()
   {
      trainingInputs = new double[numCases][numInputs];
      networkOutputs = new double[numCases];

      w1 = new double[numInputs][numHidden];
      thetaJ = new double[numHidden];
      h = new double[numHidden];
      
      w2 = new double[numHidden];

      if (training)
      {
         layer1Deltas = new double[numInputs][numHidden];
         layer2Deltas = new double[numHidden];
      }

      if (training | showGroundTruths)
         groundTruths = new double[numCases];
      
   } // allocateNetworkArrays(boolean training)

/**
* Populates the network's weight arrays either with manual values or random values.
* @param manualWeights Whether to use manual weights
*/
   public void populateNetworkArrays()
   {
      populateInputsAndTruthTable();

      if (manualWeights) // Only a valid option for a  2-2-1 network 
      {
         setPredefinedWeights();
      } // if (manualWeights)
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
      w1[0][0] = 0.9404045278126735;
      w1[0][1] = 0.2241493210468354;
      w1[1][0] = -1.0121547995672862;
      w1[1][1] = -1.432402348525032;

      w2[0] = 0.251086609938219;
      w2[1] = -0.41775164313179797;
   } // setPredefinedWeights()

/**
* Generates random weights for the network within the specified range.
*/
   public void generateRandomWeights()
   {
      for (int j = 0; j < numHidden; j++)
      {
         for (int k = 0; k < numInputs; k++)
            w1[k][j] = generateRandomDouble(min, max);

         w2[j] = generateRandomDouble(min, max);
      } // for (int j = 0; j < numHidden; j++)
   } // generateRandomWeights()

/**
 * Generates a random double within the specified range.
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @return
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
   public static double calculateError(double T, double F)
   {
      return (T - F) * (T - F) / 2.0;
   } // calculateError(double T, double F)

/**
* Calculates the activations for the hidden layer neurons.
* @param inputs The input vector to the network.
*/
   public void calculateHActivations(double[] inputs)
   {
      for (int j = 0; j < numHidden; j++)
      {
         double sum;

         sum = 0.0;
         for (int k = 0; k < numInputs; k++)
            sum += w1[k][j] * inputs[k];
         
         thetaJ[j] = sum;
         h[j] = activationFunction.apply(sum);
      } // for (int j = 0; j < numHidden; j++)
   } // calculateHActivations(double[] inputs)

/**
* Calculates the output of the network based on hidden layer activations.
*/
   public void calculateOutput()
   {
      double sum;

      sum = 0.0;
      for (int j = 0; j < numHidden; j++)
         sum += w2[j] * h[j];
      
      thetaI = sum;
      output = activationFunction.apply(sum);
   } // calculateOutput()

/**
* Performs a forward pass through the network.
* @param inputs Input vector
* @return Output of the network
*/
   public void forwardPass(double[] inputs)
   {
      calculateHActivations(inputs);
      calculateOutput();
   } // forwardPass(double[] inputs)

/**
* Runs the network on all training inputs and stores the outputs.
* @return Array of network outputs
*/
   public void run()
   {
      for (int t = 0; t < numCases; t++)
      {
         forwardPass(trainingInputs[t]);
         networkOutputs[t] = output;
      }
   } // run()

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
            double target = groundTruths[t];

            forwardPass(inputs);

            averageError += calculateError(target, output);
            train(inputs, target);
         } // for (int t = 0; t < trainingInputs.length; t++)

         averageError /= (double) numCases;
         epochs++;
      } // while (epochs < IterationMax && averageError > ECutoff)
   } // loopTraining()

/**
* Performs a single optimization (backpropagation) step to update weights.
* @param inputs Input vector
* @param hiddenLayer Activations of the hidden layer
* @param output Output of the network
* @param T Target value
*/
   public void train(double[] inputs, double T)
   {  
      double deltaOutput = -(T - output) * activationFunctionDerivative.apply(thetaI);
      for (int j = 0; j < numHidden; j++)
      {
         double deltaWeight = -learningRate * deltaOutput * h[j];
         layer2Deltas[j] = deltaWeight;
         double deltaHidden = deltaOutput * w2[j] * activationFunctionDerivative.apply(thetaJ[j]);
         
         for (int k = 0; k < numInputs; k++) 
         {
            double deltaWeightInputHidden = -learningRate * deltaHidden * inputs[k];
            layer1Deltas[k][j] = deltaWeightInputHidden;
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
            w1[k][j] += layer1Deltas[k][j];
         
         w2[j] += layer2Deltas[j];
      } // for (int j = 0; j < numHidden; j++)
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
            System.out.print("Ground Truth: " + groundTruths[t] + " ");

         System.out.println("Output: " + networkOutputs[t]);
      } // for (int t = 0; t < numCases; t++)
   } // printRunResults()

/**
 * Prints the network parameters.
* @param training Whether to include training-specific parameters
*/
   public void printNetworkParameters()
   {
      System.out.println("Network Parameters:");

      System.out.println(numInputs  + "-" + numHidden + "-1");
      System.out.println("Number of Hidden Neurons: " + numHidden);
      System.out.println("Number of Outputs: " + numOutputs);
      System.out.println("Learning Rate: " + learningRate);
      System.out.println("Weight Initialization Range: [" + min + ", " + max + "]");
      System.out.println("Activation Function: " + activationName);

      if (training) 
      {
         System.out.println("Number of Training Cases: " + numCases);
         System.out.println("Training Error Cutoff: " + ECutoff);
         System.out.println("Max Training Iterations: " + IterationMax);
      } // if (training)

      System.out.println("\nConfiguration Booleans:");
      System.out.println("Training Mode: " + training);
      System.out.println("Manual Weights: " + manualWeights);
      System.out.println("Run Test Cases: " + runTestCases);
      System.out.println("Show Inputs: " + showInputs);
      System.out.println("Show Ground Truths: " + showGroundTruths + "\n");

   } // printNetworkParameters()

/**
* Populates the input values and ground truth values for the training set.
*/
   public void populateInputsAndTruthTable()
   {
      trainingInputs[0][0] = 0.0;
      trainingInputs[0][1] = 0.0;
      trainingInputs[1][0] = 1.0;
      trainingInputs[1][1] = 0.0;
      trainingInputs[2][0] = 0.0;
      trainingInputs[2][1] = 1.0;
      trainingInputs[3][0] = 1.0;
      trainingInputs[3][1] = 1.0;

      if (training | showGroundTruths)
      {
         groundTruths[0] = 0.0;
         groundTruths[1] = 1.0;
         groundTruths[2] = 1.0;
         groundTruths[3] = 0.0;
      }
   }

/**
* Runs the network with hardcoded parameters for demonstration or testing.
*/
   public void initializeNetworkParams()
   {
      numInputs = 2;
      numHidden = 1;
      numOutputs = 1;
      learningRate = 0.3;
      numCases = 4;
      IterationMax = 100000;
      ECutoff = 0.0002;
      min = -1.5;
      max = 1.5;

      activationName = "sigmoid";
      activationFunction = ACTIVATION_MAP.get(activationName);
      activationFunctionDerivative = ACTIVATION_DERIVATIVE_MAP.get(activationName);

      training = true;
      manualWeights = false; // Only valid for a 2-2-1 network
      runTestCases = true;
      showInputs = true;
      showGroundTruths = false;
   } // initializeNetworkParams()

/**
* Main entry point for running the network.
* @param args Command-line arguments
*/
   public static void main(String[] args)
   {
      AB1Network p = new AB1Network();

      p.initializeNetworkParams();
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
         p.run();
         p.printRunResults();
      }
   } // main(String[] args)
} // class AB1Network
