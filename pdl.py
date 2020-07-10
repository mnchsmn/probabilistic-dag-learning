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
from joblib import Parallel, delayed

from softsort import SoftSort_p1

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
        log_p_roots = F.logsigmoid(self.root_probs)     # numerically stable
        p_log = torch.stack((log_p_roots, torch.log(1 - torch.exp(log_p_roots))))
        roots = gumbel_softmax(p_log, hard=True, dim=0)[0]
        self.log(f'sampled roots {roots}')
        to_sample = roots.nonzero().view(-1).tolist()  # list of nodes that will get children sampled
        # sample children
        log_p_edges = F.logsigmoid(self.edge_probs)
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
            p_log = torch.stack((log_p_edges[i,:], torch.log(1 - torch.exp(log_p_edges[i,:]))))
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

        ##make sure all nodes with children were visited, otherwise set likelihood to zero
        nodes_w_children = (torch.sum(dag,dim=1) > 0).float().view(-1,1)
        ll_edges = ll_edges * torch.prod(1-((nodes_w_children - torch.tensor(computed, dtype=torch.float)) > 0).float())
        return ll_edges

    def _get_parentless_nodes(self, dag):
        parentless = []
        for i in range(self.n_nodes):
            if torch.all(dag[:,i].squeeze() == 0):
                parentless.append(i)
        return parentless

    def _compute_likelihood_from_roots(self, dag, rootset):
        # fill roots tensor
        roots = torch.zeros(self.n_nodes)
        for i in rootset:
            roots[i] = 1
        # likelihood roots
        ll_roots = self._compute_likelihood_vector(torch.sigmoid(self.root_probs), roots)
        # likelihood rest
        ll_edges = self._compute_edge_likelihood(dag, roots)
        
        return ll_roots * ll_edges

    def likelihood(self, dag):
        parentless = self._get_parentless_nodes(dag)

        # approximation: rootset = parentless nodes, for exact calculation use likelihood_exact
        return self._compute_likelihood_from_roots(dag, parentless)

    def root_likelihood(self, dag):
        parentless = self._get_parentless_nodes(dag)
        roots = torch.zeros(self.n_nodes)
        for i in parentless:
            roots[i] = 1
        # likelihood roots
        return self._compute_likelihood_vector(torch.sigmoid(self.root_probs), roots)
        

    def likelihood_exact(self, dag):
        parentless = self._get_parentless_nodes(dag)
        # find disconnected nodes
        disconnected = []
        always_roots = []
        for i in parentless:
            if torch.sum(dag[i,:]) == 0:
                disconnected.append(i)
            else:
                always_roots.append(i)
        # compute all possible rootsets as powerset of disconnected
        disc_roots_sets = chain.from_iterable(combinations(disconnected, r) for r in range(len(disconnected)+1))
        # for each possible rootset compute likelihood and sum up
        ll = 0
        for i, disc_roots in enumerate(disc_roots_sets):
            ll += self._compute_likelihood_from_roots(dag, sorted(list(disc_roots) + always_roots))
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
        log_alpha = F.logsigmoid(self.perm_weights)
        P, _  = gumbel_sinkhorn(log_alpha, temp=self.temp, hard=True)
        P = P.squeeze()
        P_inv = P.transpose(0,1)
        log_p_edge = F.logsigmoid(self.edge_probs)                      # numerical stability
        p_log = torch.stack((log_p_edge, torch.log(1 - torch.exp(log_p_edge))))
        dag = gumbel_softmax(p_log, hard=True, dim=0)[0]

        dag = torch.matmul(torch.matmul(P, dag), P_inv)                 # permute rows & cols
        dag = dag * self.mask                                           # apply autoregressive masking
        dag = torch.matmul(torch.matmul(P_inv, dag), P)                 # permute rows & cols back to lexicographic ordering
        return dag
    
    def extract_deterministic(self, threshold):
        log_alpha = torch.log(torch.abs(self.perm_weights.detach()))
        P, _  = gumbel_sinkhorn(log_alpha, temp=1, hard=True, noise_factor=0)
        P = P.squeeze()
        P_inv = P.transpose(0,1)
        dag = (self.edge_probs.detach() > threshold).float()

        dag = torch.matmul(torch.matmul(P, dag), P_inv)                 # permute rows & cols
        dag = dag * self.mask                                           # apply autoregressive masking given by sampled order
        dag = torch.matmul(torch.matmul(P_inv, dag), P)                 # permute rows & cols back to lexicographic ordering

        return dag

class Probabilistic_DAG_Generator_TopK_SoftSort(nn.Module):

    def __init__(self, n_nodes, temp=1.0):
        """
        n_nodes: integer; number of nodes
        """
        
        super().__init__()
        self.n_nodes = n_nodes
        self.temp = temp
        self.sort = SoftSort_p1(hard=True, tau=self.temp)
        
        #Random seed
        torch.manual_seed(0)
        np.random.seed(0)

        #Mask for ordering
        self.mask = torch.triu(torch.ones(self.n_nodes, self.n_nodes),1)

        # define initial parameters
        p = torch.zeros(n_nodes, requires_grad=True)
        torch.nn.init.normal_(p)
        self.perm_weights = torch.nn.Parameter(p)
        e = torch.zeros(n_nodes, n_nodes, requires_grad=True)
        torch.nn.init.normal_(e)
        torch.diagonal(e).fill_(0)
        self.edge_probs = torch.nn.Parameter(e)
    
    def forward(self):
        logits = F.log_softmax(self.perm_weights, dim=0).view(1,-1)
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / self.temp
        P = self.sort(gumbels)
        P = P.squeeze()
        P_inv = P.transpose(0,1)
        log_p_edge = F.logsigmoid(self.edge_probs)
        p_log = torch.stack((log_p_edge, torch.log(1 - torch.exp(log_p_edge))))
        dag = gumbel_softmax(p_log, hard=True, dim=0)[0]
        
        dag = torch.matmul(torch.matmul(P, dag), P_inv)                 # permute rows & cols
        dag = dag * self.mask                                           # apply autoregressive masking given by sampled order
        dag = torch.matmul(torch.matmul(P_inv, dag), P)                 # permute rows & cols back to lexicographic ordering
        
        return dag

    def extract_deterministic(self, threshold):
        P = self.sort(self.perm_weights.detach().view(1,-1))
        P = P.squeeze()
        P_inv = P.transpose(0,1)
        dag = (self.edge_probs.detach() > threshold).float()
        print(dag.shape)
        print(P.shape)

        dag = torch.matmul(torch.matmul(P, dag), P_inv)                 # permute rows & cols
        dag = dag * self.mask                                           # apply autoregressive masking given by sampled order
        dag = torch.matmul(torch.matmul(P_inv, dag), P)                 # permute rows & cols back to lexicographic ordering

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
        model = Probabilistic_DAG_Generator_TopK_SoftSort(n_nodes)
        
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