import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.nn.functional import gumbel_softmax
from sinkhorn_ops import gumbel_sinkhorn

import random
from itertools import chain, combinations
from tqdm.auto import tqdm
import math


# ------------------------------------------------------------------------------

class Probabilistic_DAG_Generator_From_Roots(nn.Module):

    def __init__(self, n_nodes, verbose=False):
        """
        n_nodes: integer; number of nodes
        """
        
        super().__init__()
        self.n_nodes = n_nodes
        self.verbose = verbose
        
        #Random seed
        torch.manual_seed(0)
        np.random.seed(0)

        # define initial parameters
        # self.root_probs = torch.nn.Parameter(torch.rand(n_nodes, requires_grad = True))
        # self.edge_probs = torch.nn.Parameter(torch.rand(n_nodes, n_nodes, requires_grad = True))
        r = torch.zeros(n_nodes, requires_grad=True)
        torch.nn.init.normal_(r)
        self.root_probs = torch.nn.Parameter(r)
        e = torch.zeros(n_nodes, n_nodes, requires_grad=True)
        torch.nn.init.normal_(e)
        torch.diagonal(e).fill_(0)
        self.edge_probs = torch.nn.Parameter(e)
    
    def set_verbosity(self, verbose):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)

    def forward(self):
        dag = torch.zeros(self.n_nodes, self.n_nodes)   # the final dag
        sampled = np.zeros(self.n_nodes, dtype=bool)     # set of nodes that children were sampled for
        # sample roots
        p_roots = torch.sigmoid(self.root_probs)
        p = torch.stack((p_roots, 1 - p_roots))
        p_log = torch.log(p)
        roots = gumbel_softmax(p_log, hard=True, dim=0)[0]
        self.log(f'sampled roots {roots}')
        to_sample = roots.nonzero().view(-1).tolist()  # list of nodes that will get children sampled
        # sample children
        p_edges = torch.sigmoid(self.edge_probs)
        ancestors = torch.eye(self.n_nodes, dtype=torch.uint8)
        count = 0
        while(len(to_sample) > 0):
            # pick random element to sample nodes for
            i= to_sample.pop(0)
            if sampled[i]:
                continue
            self.log(f'sampling children for {i}')
            # don't sample ancestors and roots as children
            candidates = (1-ancestors[i,:].float()) * (1-roots)
            # sample children for node i
            p = torch.stack((p_edges[i,:], 1 - p_edges[i,:]))
            p_log = torch.log(p)
            dag[i,:] = gumbel_softmax(p_log, hard= True, dim=0)[0] * candidates.float()
            for j in dag[i,:].nonzero().view(-1).tolist():
                self.log(f'sampled {j}')
                # add i to ancestors of j
                ancestors[j,i] = 1
                # add all ancestors of i to j
                ancestors[j,:][ancestors[i,:]] = 1
                to_sample.append(j)
            sampled[i] = True
        return dag

    def _compute_likelihood_vector(self, probs, outcome):
        return torch.prod(torch.abs((1-outcome) - probs))

    def _compute_edge_likelihood(self, dag, roots):
        computed = np.zeros(self.n_nodes, dtype=bool)
        p_edges = torch.sigmoid(self.edge_probs)
        ancestors = torch.eye(self.n_nodes, dtype=torch.uint8)

        parents = roots.nonzero().view(-1).tolist()

        ll_edges = 1
        while(len(parents) > 0):
            # pick random element to sample nodes for
            i= parents.pop(0)
            if computed[i]:
                continue
            self.log(f'calculating likelihood of children for {i}')
            # don't sample ancestors and roots as children
            candidates = (1-ancestors[i,:].float()) * (1-roots)
            # sample children for node i
            ll = self._compute_likelihood_vector(p_edges[i,:] * candidates, dag[i,:])
            ll_edges = ll_edges * ll
            for j in dag[i,:].nonzero().view(-1).tolist():
                # add i to ancestors of j
                ancestors[j,i] = 1
                # add all ancestors of i to j
                ancestors[j,:][ancestors[i,:]] = 1
                parents.append(j)
            computed[i] = True
        return ll_edges

    def _get_parentless_nodes(self, dag):
        parentless = []
        for i in range(self.n_nodes):
            if torch.all(dag[:,i].squeeze() == 0):
                parentless.append(i)
        return parentless

    def _compute_likelihood_from_roots(self, dag, roots):
        # likelihood roots
        ll_roots = self._compute_likelihood_vector(torch.sigmoid(self.root_probs), roots)
        # likelihood rest
        ll_edges = self._compute_edge_likelihood(dag, roots)
        
        return ll_roots * ll_edges

    def likelihood(self, dag):
        parentless = self._get_parentless_nodes(dag)

        # approximation: rootset = roots, for exact calculation use likelihood_exact
        roots = torch.zeros(self.n_nodes)
        for i in parentless:
            roots[i] = 1
        return self._compute_likelihood_from_roots(dag, roots)

        

    def likelihood_exact(self, dag):
        parentless = self._get_parentless_nodes(dag)
        # compute all possible rootsets as powerset of parentless
        rootsets = chain.from_iterable(combinations(parentless, r) for r in range(len(parentless)+1))
        # for each possible rootset compute likelihood and sum up
        ll = 0
        for rootset in tqdm(rootsets, total=math.pow(2,len(parentless))):
            roots = torch.zeros(self.n_nodes)
            for i in rootset:
                roots[i] = 1
            ll += self._compute_likelihood_from_roots(dag, roots)
        return ll

        
class Probabilistic_DAG_Generator_Sinkhorn(nn.Module):

    def __init__(self, n_nodes, temp=1.0, noise_factor=1.0):
        """
        n_nodes: integer; number of nodes
        """
        
        super().__init__()
        self.n_nodes = n_nodes
        self.temp = temp
        self.noise_factor = noise_factor
        
        #Random seed
        torch.manual_seed(0)
        np.random.seed(0)

        #Mask for ordering
        self.mask = torch.triu(torch.ones(self.n_nodes, self.n_nodes),1)

        # define initial parameters
        #self.perm_weights = torch.nn.Parameter(torch.rand(n_nodes, n_nodes, requires_grad = True))
        #self.edge_probs = torch.nn.Parameter(torch.rand(n_nodes, n_nodes, requires_grad = True))
        p = torch.zeros(n_nodes, n_nodes, requires_grad=True)
        torch.nn.init.normal_(p)
        self.perm_weights = torch.nn.Parameter(p)
        e = torch.zeros(n_nodes, n_nodes, requires_grad=True)
        torch.nn.init.normal_(e)
        torch.diagonal(e).fill_(0)
        self.edge_probs = torch.nn.Parameter(e)
    
    def forward(self):
        log_alpha = torch.log(torch.sigmoid(self.perm_weights))
        P, _  = gumbel_sinkhorn(log_alpha, temp=0.1, hard=True)
        P = P.squeeze()
        P_inv = P.transpose(0,1)
        p_edge = torch.sigmoid(self.edge_probs)
        p = torch.stack((p_edge, 1 - p_edge))
        p_log = torch.log(p)
        dag = gumbel_softmax(p_log, hard=True, dim=0)[0]

        dag = torch.matmul(P, dag)                                      # permute rows
        dag = torch.matmul(P, dag.transpose(0,1)).transpose(0,1)        # permute cols
        dag = dag * self.mask                                           # apply autoregressive masking
        dag = torch.matmul(P_inv, dag)                                  # permute rows back to lexicographic ordering
        dag = torch.matmul(P_inv, dag.transpose(0,1)).transpose(0,1)    # permute cols back to lexicographic ordering
        
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
        #(10)
    ]
    
    for n_nodes in configs:
        
        print(f'checking with {n_nodes} nodes.')
        model = Probabilistic_DAG_Generator_From_Roots(n_nodes)
        
        # compute and print dag
        dag = model()
        print(dag)
        print(f'Is DAG acyclic? {is_acyclic(dag.detach().numpy())}')

        # Check gradients
        gt_graph = torch.ones(n_nodes,n_nodes)
        loss = (dag-gt_graph).sum()
        print(f'loss: {loss}')
        print(loss.grad_fn)
        loss.backward()
        try:
            print('Gradient root probs')
            print(model.root_probs.grad)
        except:
            print('Not available')
        try:
            print('Gradient edge probs')
            print(model.edge_probs.grad)
        except:
            print('Not available')
        try:
            print('Gradient perm weights')
            print(model.perm_weights.grad)
        except:
            print('Not available')
        plot_graph(graph=dag.detach().numpy().astype(int), out_path=f'test_{n_nodes}.pdf')