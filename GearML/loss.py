class BasicDistanceLoss:
    def __init__(self):
        pass
    
    def loss(self, pred, test):
        if type(pred) != list:
            return -2 * (test-pred)
        else:
            losses = []

            for i in range(len(pred)):
                losses.append([((test[i]-pred[i]))])

            new_loss = []
            for l in losses:
                for r in l:
                    new_loss.append(0)
                break

            num = 0
            for l in losses:
                num += 1
                for i in range(len(l)):
                    new_loss[i] += l[i]

            return new_loss
