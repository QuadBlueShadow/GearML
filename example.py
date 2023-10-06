from models import LinearModel
from GearML.loss import BasicDistanceLoss

loss_fn = BasicDistanceLoss()
model = LinearModel(obs=1, 
                    act=1, 
                    layer_arc=[10, 10], 
                    oo=True,
                    lr=0.001, 
                    decay=0)

states = []
tests = []
for i in range(10):
    states.append([i])
    tests.append(2*i+2)

print(model.run(states[0]), tests[0])

loss = []
for i in range(1000):
    preds = []
    for state in states:
        preds.append(model.run(state))
        
    loss = loss_fn.loss(preds, tests)

    model.adjust_weights(states, loss)

for i in range(len(states)):
    print(model.run(states[i]), tests[i])
