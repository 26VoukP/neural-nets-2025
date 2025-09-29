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
   public int numInputs, numActivations;
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
   public double[] unactivatedOutput, activations;

   /**
    * Sets the activation function and its derivative based on the provided name.
    * @param name
    */
   public void setActivationFunction(String name) 
   {
       this.activationFunction = activationMap.get(name);
       this.activationFunctionDerivative = activationDerivativeMap.get(name);
   }

   /**
    * Initializes the arrays for weights and outputs.
    * @param numActivations Number of output neurons
    * @param numInputs Number of input neurons
    */
   public void initializeArrays(int numActivations, int numInputs)
   {
      this.numInputs = numInputs;
      this.unactivatedOutput = new double[numActivations];
      this.numActivations = numActivations;
      this.activations = new double[numActivations];
      this.weights = new double[numActivations][numInputs];
   }

   /**
    * Generates a random weight between min and max.
    * @param max Maximum value
    * @param min Minimum value
    * @return Random weight
    */
   public double generateRandomWeight(double min, double max)
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
   public void initializeRandomWeights(double min, double max)
   {
      for (int i1 = 0; i1 < numActivations; i1++)
      {
         for (int i2 = 0; i2 < numInputs; i2++)
         {
           weights[i1][i2] = generateRandomWeight(min, max);
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
      for (int i = 0; i < input_activations; i++)
      {
         sum += a[i] * b[i];
      }
      return sum;
   }

   /**
    * Performs a forward pass through the layer.
    * @param inputs Input vector
    * @return Output activations
    */
   public double[] passThroughWeights(double[] inputs)
   {
      for (int i = 0; i < this.numActivations; i++)
      {
         unactivatedOutput[i] = dotProduct(inputs, this.weights[i], inputs.length);
      }
      return unactivatedOutput;
   }

   /**
    * Applies the activation function to the input values.
    * @param inputs
    * @return
    */
   public double[] activateValues(double[] unactivatedNeurons) {
       for (int i = 0; i < this.numActivations; i++) {
           activations[i] = this.activationFunction.apply(unactivatedNeurons[i]);
       }
       return activations;
   }

   /**
    * Adjusts the entire weight matrix by the given deltas.
    * @param deltas Matrix of weight changes
    */
   public void adjustWeightArray(double[][] deltas)
   {
      for (int neuronIndex = 0; neuronIndex < numActivations; neuronIndex++)
      {
         for (int weightIndex = 0; weightIndex < deltas[neuronIndex].length; weightIndex++)
         {
            this.weights[neuronIndex][weightIndex] += deltas[neuronIndex][weightIndex];
         }
      }
   }
}