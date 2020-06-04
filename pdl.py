import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.nn.functional import gumbel_softmax


# ------------------------------------------------------------------------------

class Probabilistic_DAG_Generator(nn.Module):
    def __init__(self, n_nodes):
        """
        n_nodes: integer; number of nodes
        """
        
        super().__init__()
        self.n_nodes = n_nodes
        
        #Random seed
        torch.manual_seed(0)
        np.random.seed(0)

        # define initial parameters
        self.root_probs = torch.rand(n_nodes, requires_grad = True)
        self.edge_probs = torch.rand(n_nodes, n_nodes, requires_grad = True)
        
    
    def forward(self):
        dag = torch.zeros(self.n_nodes, self.n_nodes)
        #sample roots
        roots_one_hot = torch.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            p = torch.stack((self.root_probs[i], 1-self.root_probs[i]))
            roots_one_hot[i] = gumbel_softmax(p, hard=True)[0]
        print(f'roots: {roots_one_hot}')
        #sample children
        ancestors = {k: set([k]) for k in range(self.n_nodes)}
        #TODO: the order here is a bias (-> random? learn it?)
        for i in range(self.n_nodes):
            candidates = torch.ones(self.n_nodes)
            # don't sample ancestors as children
            for k in ancestors[i]:
                candidates[k] = 0
            # don't sample roots as children 
            candidates = (1-roots_one_hot) * candidates
            # sample children for node i
            for j in range(self.n_nodes):
                p = torch.stack((self.edge_probs[i,j], 1 - self.edge_probs[i,j]))
                dag[i,j] = gumbel_softmax(p, hard= True)[0] * candidates[j]
                if dag[i,j] == 1:
                    ancestors[j].add(i)
        return dag

class Probabilistic_DAG_Generator_From_Roots(Probabilistic_DAG_Generator):
    
    def forward(self):
        dag = torch.zeros(self.n_nodes, self.n_nodes)   # the final dag
        sampled = set()                                 # set of nodes that children were sampled for
        to_sample = []                                    # list of nodes that will get children sampled
        # sample roots
        roots_one_hot = torch.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            p = torch.stack((self.root_probs[i], 1-self.root_probs[i]))
            roots_one_hot[i] = gumbel_softmax(p, hard=True)[0]
            if roots_one_hot[i] == 1:
                to_sample.append(i)
        print(f'roots: {to_sample}')
        # sample children
        ancestors = {k: set([k]) for k in range(self.n_nodes)}
        count = 0
        while(len(to_sample) > 0):
            i = to_sample.pop(0)
            if i in sampled:
                continue
            candidates = torch.ones(self.n_nodes)
            # don't sample ancestors as children
            for k in ancestors[i]:
                candidates[k] = 0
            # don't sample roots as children 
            candidates = (1-roots_one_hot) * candidates
            # sample children for node i
            # TODO: The order here is a bias!
            for j in range(self.n_nodes):
                p = torch.stack((self.edge_probs[i,j], 1 - self.edge_probs[i,j]))
                dag[i,j] = gumbel_softmax(p, hard= True)[0] * candidates[j]
                if dag[i,j] == 1:
                    ancestors[j].add(i)
                    for anc in ancestors[i]:
                        ancestors[j].add(anc)
                    to_sample.append(j)
            sampled.add(i)
        return dag

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from torch.autograd import Variable
    from utils.plot_graph import plot_graph

    def is_acyclic(adjacency):
        prod = np.eye(adjacency.shape[0])
        for _ in range(1, adjacency.shape[0] + 1):
            prod = np.matmul(adjacency, prod)
            if np.trace(prod) != 0: return False
        return True

    rng = np.random.RandomState(14)

    configs = [
        (5),                 # test various graph sizes
        (10)
    ]
    
    for n_nodes in configs:
        
        print(f'checking with {n_nodes} nodes.')
        model = Probabilistic_DAG_Generator_From_Roots(n_nodes)
        
        # compute and print dag
        dag = model()
        print(f'Is DAG acyclic? {is_acyclic(dag.detach().numpy())}')

        # Check gradients
        gt_graph = torch.ones(n_nodes,n_nodes)
        loss = (dag-gt_graph).sum()
        print(f'loss: {loss}')
        print(loss.grad_fn)
        loss.backward()
        print(model.root_probs.grad)
        print(model.edge_probs.grad)
        plot_graph(graph=dag.detach().numpy().astype(int), out_path=f'test_{n_nodes}.pdf')