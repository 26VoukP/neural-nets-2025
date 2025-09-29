/**
 * Currently, the Pereceptron class implements a simple feedforward neural network with one hidden layer.
 * It supports both manual and random weight initialization, training using gradient descent,
 * and evaluation on input data. The network can be configured for different activation functions
 * and learning rates, and is suitable for boolean operations (such as AND, OR, & EXOR).
 *
 * @author Vouk Praun-Petrovic
 * @version September 9, 2024
 */
public class Pereceptron
{
   public double min, max;
   public double learningRate, ECutoff;
   public int numInputs, numHidden, numOutputs, numLayers;
   public FeedforwardLayer layer1;
   public FeedforwardLayer layer2;
   public int numCases, IterationMax;
   public String activationName;
   public double[][] w1, w2, layer1Deltas, layer2Deltas;
   public double[][] trainingInputs;
   public double[] thetaJ, h, groundTruths, networkOutputs;
   public double thetaI, output, averageError;
   public int epochs;

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
         {
            System.out.print("Inputs: " + java.util.Arrays.toString(trainingInputs[i]) + " ");
         }
         if (includeGroundTruths)
         {
            System.out.print("Ground Truth: " + groundTruths[i] + " ");
         }
         System.out.println("Output: " + networkOutputs[i]);
      }
   }

   /**
    * Calculates the mean squared error between target and output.
    * @param T Target value
    * @param F Output value
    * @return Error value
    */
   public double calculateError(double T, double F)
   {
      return (T - F) * (T- F)/ 2;
   }

   /**
    * Runs the network with hardcoded parameters for demonstration or testing.
    */
   public void initializeNetworkParams()
   {
      numInputs = 2;
      numHidden = 5;
      numOutputs  = 1;
      layer1 = new FeedforwardLayer();
      layer2 = new FeedforwardLayer();
      learningRate = 0.3;
      numCases = 4;
      IterationMax = 100000;
      ECutoff = 0.0002;
      min = -1.5;
      max = 1.5;
      activationName = "sigmoid";
      thetaJ = new double[numHidden];
      h = new double[numHidden];
      layer1.setActivationFunction(activationName);
      layer2.setActivationFunction(activationName);
   }

   /**
    * Allocates memory for the network's arrays based on configuration.
    * @param training Whether the network is in training mode
    */
   public void allocateNetworkArrays(boolean training)
   {
      trainingInputs = new double[numCases][numInputs];
      networkOutputs = new double[numCases];
      groundTruths = new double[numCases];
      layer1.initializeArrays(numHidden, numInputs);
      layer2.initializeArrays(numOutputs, numHidden);
      if (training)
      {
         layer1Deltas = new double[numInputs][numHidden];
         layer2Deltas = new double[numHidden][numOutputs];
      }
   }

   /**
    * Populates the network's weight arrays either with manual values or random values.
    * @param MANUAL_WEIGHTS Whether to use manual weights
    */
   public void populateNetworkArrays(boolean MANUAL_WEIGHTS)
   {
      trainingInputs[0][0] = 0.0;
      trainingInputs[0][1] = 0.0;
      trainingInputs[1][0] = 1.0;
      trainingInputs[1][1] = 0.0;
      trainingInputs[2][0] = 0.0;
      trainingInputs[2][1] = 1.0;
      trainingInputs[3][0] = 1.0;
      trainingInputs[3][1] = 1.0;

      groundTruths[0] = 0.0;
      groundTruths[1] = 1.0;
      groundTruths[2] = 1.0;
      groundTruths[3] = 0.0; // XOR operation
      if (MANUAL_WEIGHTS) // Only a valid option for a  2-2-1 network 
      {
         layer1.weights = new double[][] {
            {0.9404045278126735, 0.2241493210468354},
            {-1.0121547995672862, -1.432402348525032}
         };
         layer2.weights = new double[][]{{0.251086609938219, -0.41775164313179797}};
      }
      else {
         layer1.initializeRandomWeights(min, max);
         layer2.initializeRandomWeights(min, max);
      }
   }

   public void calculateHActivations(double[] inputs)
   {
      for (int j = 0; j < numHidden; j++)
      {
         double sum = 0.0;
         for (int k = 0; k < numInputs; k++)
         {
            sum += layer1.weights[k][j] * inputs[k];
         }
         thetaJ[j] = sum;
         h[j] = layer1.activationFunction.apply(sum);
      }
   }

   public void calculateOutput()
   {
      double sum = 0.0;
      for (int j = 0; j < numHidden; j++)
      {
         sum += layer2.weights[j][0] * h[j];
      }
      thetaI = sum;
      output = layer2.activationFunction.apply(sum);
   }

   /**
    * Performs a forward pass through the network.
    * @param inputs Input vector
    * @return Output of the network
    */
   public double forwardPass(double[] inputs)
   {
      calculateHActivations(inputs);
      calculateOutput();
      return output;
   }

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
    * Trains the network for one epoch over all training cases.
    * @return Average error over all cases
    */
   public double trainNetworkOneEpoch()
   {
      averageError = 0.0;
      for (int i = 0; i < numCases; i++) 
         {
            double[] inputs = trainingInputs[i];
            double target = groundTruths[i];
            forwardPass(inputs);
            averageError += calculateError(target, output);
            optimize(inputs, target);
         } // for (int i = 0; i < trainingInputs.length; i++)
      return averageError / numCases;
   }

   /**
    * Trains the network using the provided training data.
    * @param trainingInputs Input data for training
    * @param trainingOutputs Target outputs for training
    */
   public void loopTrainingWithResults()
   {
      int epoch = 0;
      averageError = Double.MAX_VALUE;
      while (epoch < IterationMax && averageError > ECutoff)
      {
         averageError = trainNetworkOneEpoch();
         epoch++;
         if (epoch % 1000 == 0)
            System.out.println("Epoch: " + epoch + ", Average Error: " + averageError);
      } // while (epoch < IterationMax && averageError > ECutoff)
      if (epoch == IterationMax)
      {
         System.out.println("Warning: Training did not converge to desired error value within " + IterationMax + " iterations. Final error: " + averageError);
      }
      else if (averageError <= ECutoff)
      {
         System.out.println("Training converged successfully after " + epoch + " iterations. Final error: " + averageError);
      }
   }

   /**
    * Performs a single optimization (backpropagation) step to update weights.
    * @param inputs Input vector
    * @param hiddenLayer Activations of the hidden layer
    * @param output Output of the network
    * @param T Target value
    */
   public void optimize(double[] inputs, double T)
   {  
      double deltaOutput = -(T - output) * layer2.activationFunctionDerivative.apply(thetaI);
      for (int j = 0; j < numHidden; j++)
      {
         double deltaWeight = -learningRate * deltaOutput * h[j];
         layer2Deltas[j][0] = deltaWeight;
         double deltaHidden = deltaOutput * layer2.weights[j][0] * layer1.activationFunctionDerivative.apply(thetaJ[j]);
         for (int k = 0; k < numInputs; k++) 
         {
            double deltaWeightInputHidden = -learningRate * deltaHidden * inputs[k];
            layer1Deltas[k][j] = deltaWeightInputHidden;
         } // for (int k = 0; k < numInputs; k++)
      } // for (int j = 0; j < numHidden; j++)
      layer1.adjustWeightArray(layer1Deltas);
      layer2.adjustWeightArray(layer2Deltas);
   }

   /**
    * Prints the current weights of the network.
    */
   public void printNetworkWeights()
   {
      System.out.println("Weights from Input to Hidden Layer (w1):");
      System.out.println(java.util.Arrays.deepToString(layer1.weights));
      System.out.println("Weights from Hidden to Output Layer (w2):");
      System.out.println(java.util.Arrays.deepToString(layer2.weights));
   }

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
      if (training) {
         System.out.println("Number of Training Cases: " + numCases);
         System.out.println("Training Error Cutoff: " + ECutoff);
         System.out.println("Max Training Iterations: " + IterationMax);
      }
      printNetworkWeights();
   }

   /**
    * Main entry point for running the network.
    * @param args Command-line arguments
    */
   public static void main(String[] args)
   {
      boolean training = true;
      boolean manual_weights = false; // Only valid for a 2-2-1 network
      boolean runTestCases = true;
      Pereceptron p = new Pereceptron();
      p.initializeNetworkParams();
      p.allocateNetworkArrays(training);
      p.populateNetworkArrays(manual_weights);
      p.printNetworkParameters(training);
      if (training)
      {
         p.loopTrainingWithResults();
      }
      p.run();
      if (runTestCases)
      {
         boolean showInputs = true;
         boolean showGroundTruths = true;
         p.printRunResults(showInputs, showGroundTruths);
      }
   }
}