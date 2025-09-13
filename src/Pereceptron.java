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
   public String activationName;
   public int numCases, IterationMax;

   /**
    * Performs a forward pass through the network.
    * @param inputs Input vector
    * @return Output of the network
    */
   public double forwardPass(double[] inputs)
   {
      double[] hiddenActivations = layer1.passThroughLayer(inputs);
      return layer2.passThroughLayer(hiddenActivations)[0];
   }

   /**
    * Calculates the mean squared error between target and output.
    * @param T Target value
    * @param F Output value
    * @return Error value
    */
   public double calculateError(double T, double F)
   {
      return Math.pow(T - F, 2) / 2;
   }

   /**
    * Initializes the network with manually specified weights.
    * @param inputDim Number of input neurons
    * @param hiddenDim Number of hidden neurons
    * @param outputDim Number of output neurons
    * @param numExamples Number of training examples
    * @param w1 Weights from input to hidden layer
    * @param w2 Weights from hidden to output layer
    * @param activationName Activation function name
    * @param lr Learning rate
    */
   public void initializeNetwork(int inputDim, int hiddenDim, int outputDim, int numExamples, double[][] w1, double[][] w2,String activationName, double lr)
   {
      numInputs = inputDim;
      numHidden = hiddenDim;
      numOutputs = outputDim;
      learningRate = lr;
      numCases = numExamples;
      layer1 = new FeedforwardLayer();
      layer1.initializeLayer(numInputs, numHidden, w1, activationName);
      layer2 = new FeedforwardLayer();
      layer2.initializeLayer(numHidden, numOutputs, w2, activationName);
   }

   /**
    * Initializes the network with random weights.
    * @param inputDim Number of input neurons
    * @param hiddenDim Number of hidden neurons
    * @param outputDim Number of output neurons
    * @param numExamples Number of training examples
    * @param maxVal Maximum weight value
    * @param minVal Minimum weight value
    * @param activationName Activation function name
    * @param lr Learning rate
    */
   public void initializeNetwork(int inputDim, int hiddenDim, int outputDim, int numExamples, double maxVal, double minVal, String activationName, double lr)
   {
      numInputs = inputDim;
      numHidden = hiddenDim;
      numOutputs = outputDim;
      learningRate = lr;
      numCases = numExamples;
      layer1 = new FeedforwardLayer();
      layer1.initializeLayer(numInputs, numHidden, maxVal, minVal, activationName);
      layer2 = new FeedforwardLayer();
      layer2.initializeLayer(numHidden, numOutputs, maxVal, minVal, activationName);
   }

   /**
    * Runs the network with hardcoded parameters for demonstration or testing.
    */
   public void runNetworkWithHardcodedParams()
   {
      boolean MANUAL_WEIGHTS = false;
      boolean TRAINING = true;
      int inputs = 2;
      int hidden = 5;
      int outputs = 1;
      double lr = 0.01;
      if (MANUAL_WEIGHTS) {
         double[][] w1 = new double[][]{
               {0.9404045278126735, 0.2241493210468354},
               {-1.0121547995672862, -1.432402348525032},
         };
         double[][] w2 = new double[][]{{0.251086609938219, -0.41775164313179797}};
         initializeNetwork(inputs, hidden, outputs, 4, w1, w2, "linear", lr);
      } else {
         double minWeight = -1.0;
         double maxWeight = 1.0;
         initializeNetwork(inputs, hidden, outputs, 4, maxWeight, minWeight, "linear", lr);
      }
      printNetworkWeights();
      if (TRAINING) {
         double[][] trainingInputs = {
               {0.0, 0.0},
               {1.0, 0.0},
               {0.0, 1.0},
               {1.0, 1.0}
         };
         double[] trainingOutputs = {0.0, 1.0, 1.0, 1.0};
         int maxIterations = 40000;
         double errorThreshold = 0.01;
         trainNetwork(trainingInputs, trainingOutputs, maxIterations, errorThreshold);
         System.out.println("Final weights after training:");
         printNetworkWeights();
      }
      else {
         double[] input = {1.0, 0.0};
         double output = forwardPass(input);
         double error = calculateError(1.0, output);
         System.out.println("Input: " + java.util.Arrays.toString(input));
         System.out.println("Output: " + output);
         System.out.println("Error: " + error);
      }
   }

   /**
    * Trains the network using the provided training data.
    * @param trainingInputs Input data for training
    * @param trainingOutputs Target outputs for training
    * @param maxIterations Maximum number of training epochs
    * @param errorThreshold Error threshold for convergence
    */
   public void trainNetwork(double[][] trainingInputs, double[] trainingOutputs, int maxIterations, double errorThreshold)
   {
      int epoch = 0;
      double avgError = Double.MAX_VALUE;
      while (epoch < maxIterations && avgError > errorThreshold) {
         double totalError = 0.0;
         for (int i = 0; i < trainingInputs.length; i++) {
            double[] inputs = trainingInputs[i];
            double target = trainingOutputs[i];
            double[] hiddenLayer = layer1.passThroughLayer(inputs);
            double output = layer2.passThroughLayer(hiddenLayer)[0];
            totalError += calculateError(target, output);
            optimize(inputs, hiddenLayer, output, target);
         }
         epoch++;
         if (epoch % 1000 == 0) {
            avgError = totalError / numCases;
            System.out.println("Epoch: " + epoch + ", Average Error: " + avgError);
         }
      }
      if (avgError > errorThreshold) {
          System.out.println("Warning: Training did not converge to desired error value within " + maxIterations + " iterations. Final error: " + avgError);
      } else {
          System.out.println("Training converged successfully after " + epoch + " iterations. Final error: " + avgError);
      }
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
    * Performs a single optimization (backpropagation) step to update weights.
    * @param inputs Input vector
    * @param hiddenLayer Activations of the hidden layer
    * @param output Output of the network
    * @param T Target value
    */
   public void optimize(double[] inputs, double[] hiddenLayer, double output, double T)
   {
      double[][] layer1Deltas = new double[numHidden][numInputs];
      double[][] layer2Deltas = new double[numOutputs][numHidden];
      for (int f = 0; f < numOutputs; f++) {
         double deltaOutput = -(T - output) * layer2.activationFunctionDerivative.apply(output);
         for (int j = 0; j < numHidden; j++) {
            double deltaWeight = -learningRate * deltaOutput * hiddenLayer[j];
            layer2Deltas[f][j] = deltaWeight;
            for (int k = 0; k < numInputs; k++) {
               double deltaHidden = deltaOutput * layer2.weights[f][j] * layer1.activationFunctionDerivative.apply(hiddenLayer[j]);
               double deltaWeightInputHidden = -learningRate * deltaHidden * inputs[k];
               layer1Deltas[j][k] = deltaWeightInputHidden;
            }
         }
         layer1.adjustWeightArray(layer1Deltas);
         layer2.adjustWeightArray(layer2Deltas);
      }
   }

   /**
    * Main entry point for running the network.
    * @param args Command-line arguments
    */
   public static void main(String[] args)
   {
      Pereceptron p = new Pereceptron();
      p.runNetworkWithHardcodedParams();
   }
}