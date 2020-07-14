from pdl import Probabilistic_DAG_Generator_Sinkhorn
from pdl import Probabilistic_DAG_Generator_From_Roots
from pdl import Probabilistic_DAG_Generator_TopK_SoftSort

import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

import json

data_path = '../gran-dag/data'
datasets = ['data_p10_e10_n1000_GP', 'data_sf_p50_e50_n1000_GP', 'data_p50_e200_n1000_GP']
models = {
    'model_sink': Probabilistic_DAG_Generator_Sinkhorn,
    'model_roots': Probabilistic_DAG_Generator_From_Roots,
    'model_topk': Probabilistic_DAG_Generator_TopK_SoftSort
}

num_dags = 20
runs = 5
patience = 15

results = []

for dataset in datasets:
    for dag in range(1,num_dags+1):
        gt = torch.tensor(np.load(os.path.join(data_path, dataset, f'DAG{dag}.npy')), dtype=torch.float32)
        n_nodes = gt.shape[0]
        for model_name in models.keys():
            #TODO: average over runs
            for j in range(runs):
                model = models[model_name](n_nodes)
                model.seed(j)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
                model.train()
                losses = []
                best_loss = 1e30
                best_model = None
                best_epoch = -1
                patience_count = 0
                for i in range(100000):
                    out = model.forward()
                    # evaluate the loss
                    loss = F.mse_loss(out, gt, reduction='sum')
                    losses.append(loss.detach().numpy())
                    if loss.grad_fn is not None:
                        optimizer.zero_grad()
                        loss.backward()
                        if torch.isnan(model.edge_probs.grad).any():
                            print(model.edge_probs.grad)
                            break
                        #print(model.perm_weights.grad)
                        optimizer.step()
                    else:
                        pass
                    if i > 0 and i%100 == 0: #end of epoch
                        losses_arr = np.array(losses)
                        epoch_loss = losses_arr.mean()
                        print(f"Epoch {i}. Average loss: {epoch_loss}")
                        if epoch_loss < best_loss:
                            best_loss = epoch_loss
                            best_model = model
                            best_epoch = i//100
                            patience_count = 0
                        else:
                            patience_count +=1
                            if patience_count == patience:
                                print(f'Early Stopping after {i//100} epochs')
                                print(f'Best loss: {best_loss} after {best_epoch} epochs')
                                break
                        losses = []
                pred = model.extract_deterministic(0.5).detach().numpy()
                shd = (gt.numpy()-pred).sum()
                results.append({
                    'dataset': dataset,
                    'dag': dag,
                    'model': model_name,
                    'best_loss': str(best_loss),
                    'best_epoch': best_epoch,
                    'trained_epochs': i//100,
                    'shd_final': int(shd) 
                })
            # store to file after each dag
            pd.DataFrame(results).to_pickle('results.pkl')
pd.DataFrame(results).to_csv('results.csv')