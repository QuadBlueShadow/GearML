from GearML.components import Linear, Sequential, LeakyRelu, argmax, Model

class LinearModel(Model):
    def __init__(self, obs, act, layer_arc=(10, 10), act_fun=LeakyRelu(), oo=False, lr=0.01, decay=0):
        super().__init__()
        self.model = Sequential([Linear(obs, layer_arc[0])])
        self.model.append(LeakyRelu())
        
        for i in range(len(layer_arc)-1):
            self.model.append(Linear(layer_arc[i], layer_arc[i+1]))
            self.model.append(LeakyRelu())
            
        self.model.append(Linear(layer_arc[len(layer_arc)-1], act))

        self.oo = oo
        self.lr = lr
        self.activation_fun = act_fun
        self.decay = decay
        
    def run(self, x, stochastic=False):
        if self.oo:
            return self.model.run(x, stochastic)[0]
        else:
            logits = self.model.run(x, stochastic)
            return argmax(logits), logits
        
    def get_weights_biases(self):
        out = []

        for layer in self.model.layers:
            if type(layer) == Linear:
                out.append(layer.get_neruon_weights())
                
        return out