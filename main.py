

import torch

import hydra
from parameters import Parameters







if __name__ == "__main__":
    param=Parameters(epoch=10, Bertlayernumber=2)
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    best_acc = 0
    torch.manual_seed(32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    param.preparedata()

    param.creatScheduler()
    param.createOptimizer()
    param.createLoss()

    for epoch in range(param.numepoch):
        #Train
        loss, acc = param.train(epoch)
        train_accs.append(acc)
        train_losses.append(loss)
        #val
        loss, acc = param.val(epoch)
        val_accs.append(acc)
        val_losses.append(loss)
        param.scheduler.step()
        if acc>best_acc:
            best_acc = acc
            print("Best model so far")
            torch.save(param.Net, "model.pth")