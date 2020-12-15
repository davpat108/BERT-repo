

import torch

import hydra
from parameters import Parameters
from omegaconf import DictConfig, OmegaConf





@hydra.main(config_path="conf", config_name="config")
def BertMainFunction(cfg: DictConfig):
    param.configure(BatchSize=cfg.BatchSizes.BatchSize, Bertlayernumber=cfg.db.Bertlayer)
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    best_acc = 0
    torch.manual_seed(32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
            f=open("log.txt", "a")
            f.write("RUN: BatchSize"+str(cfg.BatchSizes.BatchSize)+ " BertLayer"+ str(cfg.db.Bertlayer)+str(acc)+"\n")
            f.close()



if __name__ == "__main__":
    param=Parameters()
    BertMainFunction()