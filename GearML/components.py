import random
from GearML.Math import *

class LeakyRelu:
    def __init__(self, degrade=0.01):
        self.degrade = degrade

    def run(self, x, s=None):
        for i in range(len(x)):
            if x[i] < 0:
               x[i] = x[i] * self.degrade
               
        return x
    
    def der(self, x):
        if x > 0:
            return 1
        else:
            return 0.01

def argmax(x): 
    num = max(x)
        
    for i in range(len(x)):
        if x[i] == num:
            return i

class Argmax:
    def __init__(self):
        pass
    
    def run(self, x, s=None):
        num = max(x)
        
        for i in range(len(x)):
            if x[i] == num:
                return i

class Neuron:
    def __init__(self, next_l=1, rt=150):
        self.weights = []
        self.random_t = rt
        self.input = 0
        
        for i in range(next_l):
            weight = 0
            if random.random() > 0.5:
                weight = random.random()
            else:
                weight = -random.random()

            self.weights.append(weight)
        
    def run(self, input, stochastic):
        outputs = []
        self.input = input

        for i in range(len(self.weights)):
            r = 0
            if stochastic:
                r = random.randint(-self.random_t, self.random_t)
                r /= 10
                
            outputs.append((input*self.weights[i])+r)
            
        return outputs
        
    def get_weights(self):
        return self.weights
        
class Linear:
    def __init__(self, num_neurons=1, next_layer=1, randomness_threshold=150):
        self.neurons = []
        self.biases = []
        
        for i in range(num_neurons):
            new_neuron = Neuron(next_layer, randomness_threshold)
            self.neurons.append(new_neuron)

        for i in range(next_layer):
            self.biases.append(random.randint(-3, 3))
            
    def run(self, inputs=[], stochastic=False):
        outputs = []
        
        for i in range(len(self.neurons)):
            outputs.append(self.neurons[i].run(inputs[i], stochastic))
            
        total_output = []
        for i in range(len(outputs)):
            output = outputs[i]
            for x in range(len(output)):
                try:
                    total_output[x] += output[x]
                except:
                    total_output.append(output[x])

        for i in range(len(total_output)):
            total_output[i] += self.biases[i]
                    
        return total_output
        
    def get_neruon_weights(self):
        neuron_info = []
        for neuron in self.neurons:
            neuron_info.append(neuron.get_weights())
            
        return neuron_info
    
    def adjust_randomness(self, val=-1):
        for i in range(len(self.neurons)):
            self.neurons[i].rt += val
        
class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers
        
    def append(self, layer):
        self.layers.append(layer)
        
    def run(self, x, stochastic=False):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i].run(out, stochastic)
            
        return out
    
class Model:
    def __init__(self):
        self.model:Sequential = None
        self.lr = 0.01
        self.oo = False
        self.activation_fun = LeakyRelu()
        self.decay = 0

    def adjust_weights(self, states, loss):
        #self.model.run(states[i])
        #print(losses[i])
        self.loss = loss

        for x in range(len(self.model.layers) - 1, -1, -1):
            layer = self.model.layers[x]
            bias_mults = None
            
            if type(layer) == Linear:
                if x != len(self.model.layers) - 1:
                    bias_mults = self.look_through_sequence(x, True)

                for w in range(len(layer.biases)):
                    if x == len(self.model.layers) - 1:
                        layer.biases[w] += loss[w] * self.lr
                    else:
                        layer.biases[w] += self.lr * bias_mults[w]

                weight_mults = self.look_through_sequence(x, False)
                #print(weight_mults)

                for w in range(len(layer.neurons)):
                    neuron = layer.neurons[w]
                    for z in range(len(neuron.weights)):
                        if x == len(self.model.layers)-1:
                            neuron.weights[z] += weight_mults[z][0] * self.lr
                        else:
                            neuron.weights[z] += weight_mults[z] * self.lr

        if self.lr > self.decay:
            self.lr -= self.decay
                
    def look_through_sequence(self, layer_index, bias=True, og=True):
        if bias:
            n_layer_weights = []
            weights = []
            bias_mults = []

            layer = self.model.layers[layer_index]

            if layer_index != len(self.model.layers) - 1:
                if layer_index < len(self.model.layers) - 1:
                    n_layer_weights = self.look_through_sequence(layer_index + 2, bias, False)
                for x in range(len(layer.neurons)):
                    neuron = layer.neurons[x]
        
                    if not og:
                        t_weights = []
                        for z in range(len(neuron.weights)):
                            c_weights = n_layer_weights[z]
                
                            for weight in c_weights:
                                t_weights.append(neuron.weights[z] * weight)
                
                        weights.append(t_weights)

            if og:
                for c in range(len(n_layer_weights)):
                    weights = n_layer_weights[c]
                    mult = 0
                    for weight in weights:
                        mult += weight

                    if mult == 0:
                        mult = 1

                    bias_mults.append(mult)

            if len(bias_mults) > 0:
                return bias_mults
            elif len(weights) > 0:
                return weights
            else:
                weights = []
                n = self.model.layers[-1].neurons
                for i in range(len(n)):
                    neuron = []
                    for x in range(len(n[i].weights)):
                        neuron.append(n[i].weights[x] * self.loss[x])
                    weights.append(neuron)
                return weights
        else:
            n_layer_weights = []
            weights = []
            weight_mults = []

            layer = self.model.layers[layer_index]

            if layer_index != len(self.model.layers) - 1:
                if layer_index < len(self.model.layers) - 1:
                    n_layer_weights = self.look_through_sequence(layer_index + 2, bias, False)
                for x in range(len(layer.neurons)):
                    neuron = layer.neurons[x]

                    if not og:
                        t_weights = []
                        for z in range(len(neuron.weights)):
                            c_weights = n_layer_weights[z]

                            for weight in c_weights:
                                t_weights.append(neuron.weights[z] * weight)

                            weights.append(t_weights)

            if og:
                for c in range(len(n_layer_weights)):
                    weights = n_layer_weights[c]
                    mult = 0
                    for weight in weights:
                        mult += weight

                    if mult == 0:
                        mult = 1

                    weight_mults.append(mult)

            if len(weight_mults) > 0:
                return weight_mults
            elif len(weights) > 0:
                return weights
            else:
                weights = []
                n = self.model.layers[-1].neurons
                for i in range(len(n)):
                    neuron = []
                    for x in range(len(n[i].weights)):
                        neuron.append(n[i].weights[x] * self.loss[x])
                    weights.append(neuron)
                return weights
