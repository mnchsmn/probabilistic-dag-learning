import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_graph(graph, out_path):
    G = nx.from_numpy_matrix(graph, create_using=nx.DiGraph())

    fig = plt.figure(figsize=(15, 6))
    positions = nx.circular_layout(G)

    nx.draw_networkx(G, with_labels=True, width=2, pos=positions)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_gt_vs_pred(gt_graph, pred_graph, out_path=None):
    G_gt = nx.from_numpy_matrix(gt_graph, create_using=nx.DiGraph())
    Skeleton_gt = nx.from_numpy_matrix(gt_graph).to_directed()
    G_pred = nx.from_numpy_matrix(pred_graph, create_using=nx.DiGraph())
    
    fig, axs = plt.subplots(1,2, figsize=(15, 6))
    positions = nx.circular_layout(G_pred)

    tp = nx.intersection(G_pred, G_gt)
    r = nx.difference(nx.intersection(Skeleton_gt, G_pred), tp)
    fp = nx.difference(nx.difference(G_pred, G_gt), r.to_undirected().to_directed())
    fn = nx.difference(nx.difference(G_gt, G_pred), r.to_undirected().to_directed())

    # Ground Truth
    axs[0].title.set_text('Ground Truth DAG')
    nx.draw_networkx(G_gt, with_labels=True, width =2, ax=axs[0], pos=positions)

    # Predicted
    axs[1].title.set_text('Predicted DAG')
    nx.draw_networkx_nodes(G_pred, ax=axs[1], pos=positions)
    nx.draw_networkx_labels(G_pred, ax=axs[1], pos=positions)
    nx.draw_networkx_edges(tp, edge_color='tab:green', width=2, with_labels=True, ax=axs[1], pos=positions)
    nx.draw_networkx_edges(r, edge_color='tab:blue', width=2, ax=axs[1], pos=positions)
    nx.draw_networkx_edges(fp, edge_color='tab:red', width=2, ax=axs[1], pos=positions)
    fn_lines = nx.draw_networkx_edges(fn, edge_color='tab:orange', width=2, ax=axs[1], pos=positions)
    # Bug workaround for dashed line:
    if fn_lines is not None:
        for patch in fn_lines:
            patch.set_linestyle('dashed')


    #Legend
    l_tp = mpatches.Patch(color='green', label='True positive edges')
    l_r = mpatches.Patch(color='tab:blue', label='Reverse edges')
    l_fp = mpatches.Patch(color='tab:red', label='False positive edges')
    l_fn = mpatches.Patch(color='tab:orange', label='False negative edges')

    plt.legend(handles=[l_tp, l_r, l_fp, l_fn], bbox_to_anchor=(0,0,1,1), loc=(-0.5,0.8))

    plt.subplots_adjust(wspace=0.5)
    for ax in axs:
        ax.set_axis_off()
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()
    

def plot_gt_vs_pred_from_files(gt_path, pred_path, out_path):
    gt = np.load(gt_path)
    pred = np.load(pred_path)
    plot_gt_vs_pred(gt, pred)
    