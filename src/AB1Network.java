/**
 * The AB1Network class implements a simple feedforward neural network with one hidden layer.
 * It supports both manual and random weight initialization, training using gradient descent,
 * and evaluation on input data. The network can be configured for different activation functions
 * and learning rates and can undergo supervised learning on a predefined dataset.
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
   public String activationName;
   public Function activationFunction, activationFunctionDerivative;
   public double[][] w1, w1Deltas;
   public double[] w2, w2Deltas;
   public double[][] trainingInputs;
   public double[] groundTruths, networkOutputs;
   public double[] theta, h;
   public double thetaZero, output;
   public double averageError;
   public int epoch, epochReportingInterval;

   /**
    * Functional interface for activation functions and their derivatives.
    */
   @FunctionalInterface
   public interface Function
   {
      double apply(double x);
   }
   /**
    * Available list of activation functions for the network's layers.
    */
   public static final java.util.HashMap<String, Function> ACTIVATION_MAP = new java.util.HashMap<>();
   static
   {
      ACTIVATION_MAP.put("sigmoid", x -> 1 / (1 + Math.exp(-x)));
      ACTIVATION_MAP.put("linear", x -> x);
   }
   /**
    * Corresponding derivatives for all options of activation functions. 
    */
   public static final java.util.HashMap<String, Function> ACTIVATION_DERIV_MAP = new java.util.HashMap<>();
   static
   {
      ACTIVATION_DERIV_MAP.put("sigmoid", x -> {
         double fx = ACTIVATION_MAP.get("sigmoid").apply(x);
         return fx * (1 - fx);
      });
      ACTIVATION_DERIV_MAP.put("linear", x -> 1.0); // derivative of a line with slope 1.0 is 1.0
   }

   /**
    * Generates a random weight between min and max.
    * @param max Maximum value
    * @param min Minimum value
    * @return Random weight
    */
   public static double generateRandomWeight(double min, double max)
   {
      return Math.random() * (max - min) + min;
   } // generateRandomWeight(double min, double max)

   /**
    * Initializes the weight matrix with random values.
    * @param max Maximum weight value
    * @param min Minimum weight value
    */
   public void initializeRandomWeights()
   {
      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            w1[k][j] = generateRandomWeight(min, max);
            w2[j] = generateRandomWeight(min, max); // Assumes single output neuron
         }
      }
   } // initializeRandomWeights(double min, double max)

   /**
    * Takes the dot product of the input and the layer 1 weights.
    * @param input the values inputted to the network
    * @return the unactivated hidden neurons
    */
   public void passThroughW1(double[] input)
   {
      double sum;

      for (int j = 0; j < numHidden; j++)
      {
         sum = 0.0;
         for (int k = 0; k < numInputs; k++)
         {
            sum += input[k] * w1[k][j];
         }
         theta[j] = sum;
      }
   }

   /**
    * Performs a forward pass through the network.
    * @param inputs Input vector
    * @return Output of the network
    */
   public double forwardPass(double[] inputs)
   {
      passThroughW1(inputs);
      activateHiddenValues();
      passThroughW2();
      return activateOutput();
   } // forwardPass(double[] inputs)

   /**
    * Runs the network on all training inputs and stores the outputs.
    * @return Array of network outputs
    */
   public void run()
   {
      for (int i = 0; i < numCases; i++)
      {
         networkOutputs[i] = forwardPass(trainingInputs[i]);
      }
   }

   /**
    * Prints the results of the network run.
    * @param includeInputs Whether to include input values
    * @param includeGroundTruths Whether to include ground truth values
    */
   public void printRunResults(boolean includeInputs, boolean includeGroundTruths)
   {
      System.out.println("Run Results:");
      for (int i = 0; i < numCases; i++)
      {
         if (includeInputs)
            System.out.print("Inputs: " + java.util.Arrays.toString(trainingInputs[i]) + " ");
         
         if (includeGroundTruths)
            System.out.print("Ground Truth: " + groundTruths[i] + " ");

         System.out.println("Output: " + networkOutputs[i]);
      }
   } // printRunResults(boolean includeInputs, boolean includeGroundTruths)

   /**
    * Calculates the mean squared error between target and output.
    * @param T Target value
    * @param F Output value
    * @return Error value
    */
   public static double calculateError(double T, double F)
   {
      return (T - F) * (T - F) / 2.0;
   }

   /**
    * Runs the network with hardcoded parameters for demonstration or testing.
    */
   public void initializeNetworkParams()
   {
      numInputs = 2;
      numHidden = 5;
      numOutputs  = 1;
      learningRate = 0.3;
      numCases = 4;
      IterationMax = 100000;
      ECutoff = 0.0002;
      epochReportingInterval = 1000;
      min = -1.5;
      max = 1.5;

      activationName = "sigmoid";
      activationFunction = ACTIVATION_MAP.get(activationName);
      activationFunctionDerivative = ACTIVATION_DERIV_MAP.get(activationName);
   } // initializeNetworkParams()

   /**
    * Allocates memory for the network's arrays based on configuration.
    * @param training Whether the network is in training mode
    */
   public void allocateNetworkArrays(boolean training, boolean showGroundTruths)
   {
      trainingInputs = new double[numCases][numInputs];
      networkOutputs = new double[numCases];

      w1 = new double[numInputs][numHidden];
      theta = new double[numHidden];
      h = new double[numHidden];

      w2 = new double[numHidden];
      if (training)
      {
         w1Deltas = new double[numInputs][numHidden];
         w2Deltas = new double[numHidden];
         groundTruths = new double[numCases];
      }
      else if (showGroundTruths)
      {
         groundTruths = new double[numCases];
      }
   } // allocateNetworkArrays(boolean training, boolean showGroundTruths)

   /**
    * Populates the network's weight arrays either with manual values or random values.
    * @param MANUAL_WEIGHTS Whether to use manual weights
    */
   public void populateNetworkArrays(boolean MANUAL_WEIGHTS)
   {
      trainingInputs[0][0] = 0.0;
      trainingInputs[0][1] = 0.0;
      trainingInputs[1][0] = 0.0;
      trainingInputs[1][1] = 1.0;
      trainingInputs[2][0] = 1.0;
      trainingInputs[2][1] = 0.0;
      trainingInputs[3][0] = 1.0;
      trainingInputs[3][1] = 1.0;

      groundTruths[0] = 0.0;
      groundTruths[1] = 1.0;
      groundTruths[2] = 1.0;
      groundTruths[3] = 0.0;
      if (MANUAL_WEIGHTS) // Only a valid option for a  2-2-1 network
      {
         w1[0][0] = 0.9404045278;
         w1[0][1] = -1.0121548;
         w1[1][0] = 0.224149321;
         w1[1][1] = -1.432402349;
         w2[0] = 0.2510866099;
         w2[1] = -0.4177516431;
      }
      else 
      {
         initializeRandomWeights();
      }
   } // populateNetworkArrays(boolean MANUAL_WEIGHTS)

   /**
    * Takes the dot product of the activated hidden layer values
    * and the layer 2 weights.
    * @return the unactivated output of the network
    */
   public void passThroughW2()
   {
      thetaZero = 0.0;
      for (int j = 0; j < numHidden; j++)
      {
         thetaZero += h[j] * w2[j]; // Assumes single output neuron
      }
   }

   /**
    * Applies the activation function tp the neural network's hidden neurons using the network's activation function.
    * @return the activated neurons.
    */
   public void activateHiddenValues()
   {
      for (int j = 0; j < numHidden; j++)
      {
         h[j] = activationFunction.apply(theta[j]);
      }
   }

   /**
    * Applies the activation function to the output value of the network.
    * @return the network's prediction of the ground truth.
    */
   public double activateOutput()
   {
      output = activationFunction.apply(thetaZero);
      return output;
   }

   /**
    * Adjusts the weight arrays using the computed deltas.
    */
   public void applyWeightDeltas()
   {
      for (int k = 0; k < numInputs; k++)
         for (int j = 0; j < numHidden; j++)
         {
            w1[k][j] += w1Deltas[k][j];
            w2[j] += w2Deltas[j]; // Assumes single output neuron
         }
      
   } // applyWeightDeltas()

   /**
    * Performs a single optimization (backpropagation) step to update weights.
    * @param inputs Input vector
    * @param hiddenLayer Activations of the hidden layer
    * @param output Output of the network
    * @param T Target value
    */
   public void train(double[] inputs, double T)
   {
      double omegaZero, psiZero, partialDerivEWj, deltaWj, psiJ, partialDerivWkj, deltaWkj, omegaJ;

      omegaZero = (T - output);
      psiZero = omegaZero * activationFunctionDerivative.apply(thetaZero);
      for (int j = 0; j < numHidden; j++)
      {
         partialDerivEWj = -h[j] * psiZero;
         deltaWj = -learningRate * partialDerivEWj;
         w2Deltas[j] = deltaWj;

         omegaJ = psiZero * w2[j];
         psiJ = omegaJ * activationFunctionDerivative.apply(theta[j]);
         for (int k = 0; k < numInputs; k++) 
         {
            partialDerivWkj = -inputs[k] * psiJ;
            deltaWkj = -learningRate * partialDerivWkj;
            w1Deltas[k][j] = deltaWkj;
         } // for (int k = 0; k < numInputs; k++)
      } // for (int j = 0; j < numHidden; j++)

      applyWeightDeltas();
   } // train(double[] inputs, double T)

   /**
    * Trains the network using the provided training data.
    * @param trainingInputs Input data for training
    * @param trainingOutputs Target outputs for training
    */
   public void loopTraining()
   {
      double target;
      
      epoch = 0;
      averageError = Double.MAX_VALUE;
      while (epoch < IterationMax && averageError > ECutoff)
      {
         averageError = 0.0;
         for (int t = 0; t < numCases; t++) 
         {
            target = groundTruths[t];
            passThroughW1(trainingInputs[t]);
            activateHiddenValues();
            passThroughW2();
            activateOutput();
            averageError += calculateError(target, output);
            train(trainingInputs[t], target);
         } // for (int t = 0; t < numCases; t++)

         averageError /= numCases;
         epoch++;
         if (epoch % epochReportingInterval == 0)
            System.out.println("Epoch: " + epoch + ", Average Error: " + averageError);
      } // while (epoch < IterationMax && averageError > ECutoff)
   } // loopTraining()

   /**
    * Prints the results of the training process.
    */
   public void printTrainingResults()
   {
      if (epoch == IterationMax)
         System.out.println("Warning: Training did not converge to desired error value within "
                            + IterationMax + " iterations. Final error: " + averageError);
      else if (averageError <= ECutoff)
         System.out.println("Training converged successfully after " + epoch + " iterations. Final error: " + averageError);
   } // printTrainingResults()

   /**
    * Prints the current weights of the network.
    */
   public void printNetworkWeights()
   {
      System.out.println("Weights from Input to Hidden Layer (w1):");
      System.out.println(java.util.Arrays.deepToString(w1));
      System.out.println("Weights from Hidden to Output Layer (w2):");
      System.out.println(java.util.Arrays.toString(w2));
   } // printNetworkWeights()

   /**
    * Prints the network parameters.
    * @param training Whether to include training-specific parameters
    */
   public void printNetworkParameters(boolean training)
   {
      System.out.println("Network Parameters:");
      System.out.println("Number of Inputs: " + numInputs);
      System.out.println("Number of Hidden Neurons: " + numHidden);
      System.out.println("Number of Outputs: " + numOutputs);
      System.out.println("Learning Rate: " + learningRate);
      System.out.println("Activation Function: " + activationName);
      System.out.println("Weight Initialization Range: [" + min + ", " + max + "]");

      if (training) {
         System.out.println("Number of Training Cases: " + numCases);
         System.out.println("Training Error Cutoff: " + ECutoff);
         System.out.println("Max Training Iterations: " + IterationMax);
      }
      
      printNetworkWeights();
   } // printNetworkParameters(boolean training)

   /**
    * Main entry point for running the network.
    * @param args Command-line arguments
    */
   public static void main(String[] args)
   {
      boolean training = true;
      boolean showGroundTruths = true;
      boolean manual_weights = false; // Only valid for a 2-2-1 network
      boolean runTestCases = true;
      AB1Network p = new AB1Network();

      p.initializeNetworkParams();
      p.allocateNetworkArrays(training, showGroundTruths);
      p.populateNetworkArrays(manual_weights);
      p.printNetworkParameters(training);
      if (training)
      {
         p.loopTraining();
         p.printTrainingResults();
      }

      if (runTestCases || !training)
      {
         boolean showInputs = true;
         p.run();
         p.printRunResults(showInputs, showGroundTruths);
      }
   } // main(String[] args)

} // public class AB1Network