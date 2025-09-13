/**
 * The FeedforwardLayer class represents a single layer in a feedforward neural network.
 * It supports customizable activation functions, random or manual weight initialization,
 * and provides methods for forward propagation and weight adjustment.
 *
 * @author Vouk Praun-Petrovic
 * @version Septempber 11, 2024
 */
public class FeedforwardLayer
{
   public int numActivations;
   public static final java.util.HashMap<String, Function> activationMap = new java.util.HashMap<>();
   static
   {
      activationMap.put("sigmoid", x -> 1 / (1 + Math.exp(-x)));
      activationMap.put("relu", x -> Math.max(0, x));
      activationMap.put("linear", x -> x);
   }
   public static final java.util.HashMap<String, Function> activationDerivativeMap = new java.util.HashMap<>();
   static
   {
      activationDerivativeMap.put("sigmoid", x -> {
         double fx = activationMap.get("sigmoid").apply(x);
         return fx * (1 - fx);
      });
      activationDerivativeMap.put("relu", x -> x > 0 ? 1 : 0);
      activationDerivativeMap.put("linear", x -> 1.0);
   }
   /**
    * Functional interface for activation functions and their derivatives.
    */
   @FunctionalInterface
   public interface Function
   {
      double apply(double x);
   }
   public Function activationFunction;
   public Function activationFunctionDerivative;
   public double[][] weights;
   /**
    * Initializes the layer with random weights and a specified activation function.
    * @param inputValues Number of input neurons
    * @param numActivations Number of output neurons
    * @param maxWeight Maximum weight value
    * @param minWeight Minimum weight value
    * @param activationFunction Name of the activation function
    */
   public void initializeLayer(int inputValues, int numActivations, double maxWeight, double minWeight, String activationFunction)
   {
      this.numActivations = numActivations;
      this.activationFunction = activationMap.getOrDefault(activationFunction.toLowerCase(), activationMap.get("linear"));
      this.activationFunctionDerivative = activationDerivativeMap.getOrDefault(activationFunction.toLowerCase(), activationDerivativeMap.get("linear"));
      initializeRandomWeights(maxWeight, minWeight, numActivations, inputValues);
   }

   /**
    * Initializes the layer with manually specified weights and activation function.
    * @param inputValues Number of input neurons
    * @param numActivations Number of output neurons
    * @param weights Weight matrix
    * @param activationFunction Name of the activation function
    */
   public void initializeLayer(int inputValues, int numActivations, double[][] weights, String activationFunction)
   {
      this.numActivations = numActivations;
      this.activationFunction = activationMap.getOrDefault(activationFunction.toLowerCase(), activationMap.get("linear"));
      this.activationFunctionDerivative = activationDerivativeMap.getOrDefault(activationFunction.toLowerCase(), activationDerivativeMap.get("linear"));
      this.weights = weights;
   }

   /**
    * Generates a random weight between min and max.
    * @param max Maximum value
    * @param min Minimum value
    * @return Random weight
    */
   public double generateRandomWeight(double max, double min)
   {
      return Math.random() * (max - min) + min;
   }

   /**
    * Initializes the weight matrix with random values.
    * @param max Maximum weight value
    * @param min Minimum weight value
    * @param activations Number of output neurons
    * @param inputs Number of input neurons
    */
   public void initializeRandomWeights(double max, double min, int activations, int inputs)
   {
      this.weights = new double[activations][inputs];
      for (int i1 = 0; i1 < activations; i1++) {
         for (int i2 = 0; i2 < inputs; i2++) {
           weights[i1][i2] = generateRandomWeight(max, min);
         }
      }
   }

   /**
    * Computes the dot product of two vectors up to a given length.
    * @param a First vector
    * @param b Second vector
    * @param input_activations Number of elements to use
    * @return Dot product
    */
   public double dotProduct(double[] a, double[] b, int input_activations)
   {
      double sum = 0.0;
      for (int i = 0; i < input_activations; i++) {
         sum += a[i] * b[i];
      }
      return sum;
   }

   /**
    * Performs a forward pass through the layer.
    * @param inputs Input vector
    * @return Output activations
    */
   public double[] passThroughLayer(double[] inputs)
   {
      double[] hidden = new double[this.numActivations];
      for (int i = 0; i < this.numActivations; i++) {
         hidden[i] = this.activationFunction.apply(dotProduct(inputs, this.weights[i], inputs.length));
      }
      return hidden;
   }

   /**
    * Adjusts a single weight by a given delta.
    * @param neuronIndex Index of the neuron
    * @param weightIndex Index of the weight
    * @param delta Value to add to the weight
    */
   public void adjustWeight(int neuronIndex, int weightIndex, double delta)
   {
      this.weights[neuronIndex][weightIndex] += delta;
   }

   /**
    * Adjusts the entire weight matrix by the given deltas.
    * @param deltas Matrix of weight changes
    */
   public void adjustWeightArray(double[][] deltas)
   {
      for (int neuronIndex = 0; neuronIndex < numActivations; neuronIndex++) {
         for (int weightIndex = 0; weightIndex < deltas[neuronIndex].length; weightIndex++) {
            this.weights[neuronIndex][weightIndex] += deltas[neuronIndex][weightIndex];
         }
      }
   }
}