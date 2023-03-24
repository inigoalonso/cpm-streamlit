"""Helpers for the app."""
import streamlit as st
import copy
import random
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle


def ppm(aMatrix: list):
    '''
    Prints a numeric matrix using three decimals.

        Parameters:
            aMatrix (list):  The matrix (list of lists) to display.
        
        Returns:
            none
    '''
    st.write('\n'.join(['\t'.join([str("{:.3f}".format(cell)) for cell in row]) for row in aMatrix]))


def propagation_tree(graph,target,source):

    propagation_tree = nx.DiGraph()

    for path in list(nx.all_simple_paths(graph, source, target)):
        new_path = []
        for level, node in enumerate(path, 1):
            node_id = ""
            for i in range(level):
                if i == 0:
                    node_id = str(path[i])
                else:
                    node_id = node_id + "-" + str(path[i])
            node_name = node
            node_level = level
            new_path.append(node_id)
            propagation_tree.add_node(node_id, name=node_name, level=node_level)
        nx.add_path(propagation_tree, new_path)
    
    return propagation_tree


def combined_likelihood_matrix(DSM,direct_likelihood_matrix):
    '''Returns the Combined likelihood matrix (L)'''
    g = nx.from_numpy_array(np.transpose(np.matrix(DSM)), create_using=nx.DiGraph)
    combined_likelihood_matrix = [[0 for col in range(len(DSM))] for row in range(len(DSM))]
    for target, row in enumerate(DSM):
        #print(row)
        for source, element in enumerate(row):
            if target != source:
                #print(f"Change source: {source} Change target: {target}")
                tree = propagation_tree(g,target,source)
                nodes_list = list(reversed(sorted(tree.nodes, key=len)))
                for node in nodes_list:
                    parents = list(tree.predecessors(node))
                    children = list(tree.successors(node))
                    if len(parents) > 0:
                        parent = parents[0]
                        #print(f"Node: {node}  Data: {tree.nodes[node]} Parent: {parent} Children: {children}")
                        if len(children) == 0:
                            tree.edges[parent,node]['likelihood'] = direct_likelihood_matrix[tree.nodes[node]['name']][tree.nodes[parent]['name']]
                        elif len(children) == 1:
                            child = children[0]
                            tree.edges[parent,node]['likelihood'] = direct_likelihood_matrix[tree.nodes[node]['name']][tree.nodes[parent]['name']] * tree.edges[node,child]['likelihood']
                        else:
                            temp = 1
                            for child in children:
                                temp = temp * (1 - tree.edges[node,child]['likelihood'])
                            tree.edges[parent,node]['likelihood'] = 1 - temp
                        #print(f"Likelihood: {tree.edges[parent,node]['likelihood']}")
                    else:
                        #print(f"Node: {node}  Data: {tree.nodes[node]} Children: {children}")
                        #print(f"Root node reached")
                        temp = 1
                        for child in children:
                            temp = temp * (1 - tree.edges[node,child]['likelihood'])
                        likelihood = 1 - temp
                        #print(f"Likelihood_{source},{target}: {likelihood}")
                        combined_likelihood_matrix[target][source] = likelihood
    return combined_likelihood_matrix


#TODO combined risk matrix is for now a copy of the combined likelihood matrix function
def combined_risk_matrix(DSM,direct_likelihood_matrix,direct_impact_matrix):
    '''Returns the Combined risk matrix (R)'''
    g = nx.from_numpy_array(np.transpose(np.matrix(DSM)), create_using=nx.DiGraph)
    combined_likelihood_matrix = [[0 for col in range(len(DSM))] for row in range(len(DSM))]
    for target, row in enumerate(DSM):
        #print(row)
        for source, element in enumerate(row):
            if target != source:
                #print(f"Change source: {source} Change target: {target}")
                tree = propagation_tree(g,target,source)
                nodes_list = list(reversed(sorted(tree.nodes, key=len)))
                for node in nodes_list:
                    parents = list(tree.predecessors(node))
                    children = list(tree.successors(node))
                    if len(parents) > 0:
                        parent = parents[0]
                        #print(f"Node: {node}  Data: {tree.nodes[node]} Parent: {parent} Children: {children}")
                        if len(children) == 0:
                            tree.edges[parent,node]['likelihood'] = direct_likelihood_matrix[tree.nodes[node]['name']][tree.nodes[parent]['name']]
                        elif len(children) == 1:
                            child = children[0]
                            tree.edges[parent,node]['likelihood'] = direct_likelihood_matrix[tree.nodes[node]['name']][tree.nodes[parent]['name']] * tree.edges[node,child]['likelihood']
                        else:
                            temp = 1
                            for child in children:
                                temp = temp * (1 - tree.edges[node,child]['likelihood'])
                            tree.edges[parent,node]['likelihood'] = 1 - temp
                        #print(f"Likelihood: {tree.edges[parent,node]['likelihood']}")
                    else:
                        #print(f"Node: {node}  Data: {tree.nodes[node]} Children: {children}")
                        #print(f"Root node reached")
                        temp = 1
                        for child in children:
                            temp = temp * (1 - tree.edges[node,child]['likelihood'])
                        likelihood = 1 - temp
                        #print(f"Likelihood_{source},{target}: {likelihood}")
                        combined_likelihood_matrix[target][source] = likelihood
    return combined_likelihood_matrix


def combined_impact_matrix(DSM,clm,crm):
    '''Returns the Combined impact matrix (I)'''
    cim = [[0 for col in range(len(DSM))] for row in range(len(DSM))]

    for i, row in enumerate(crm):
        for j,element in enumerate(row):
            if i != j:
                cim[i][j] = crm[i][j]/clm[i][j]
                # print(f'{crm[i][j]}/{clm[i][j]}={cim[i][j]}')
    return cim


#TODO fix labels so the show in between ticks
def plot_product_risk_matrix(product_components,DSM,clm,cim,crm):

    n = len(DSM)
    x = []
    y = []
    for i in range(n):
        for j in range(n):
            x.append(j)
            y.append(i)

    dx = [item for sublist in clm for item in sublist]
    dy = [item for sublist in cim for item in sublist]
    z = [item for sublist in crm for item in sublist]

    cmap = plt.cm.coolwarm
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, aspect='equal')

    # Paint the grid for all elements
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    # Tick labels
    ax.set_xticklabels(product_components)
    ax.set_yticklabels(product_components)
    # for label in ax.get_xticklabels():
    #     label.set_horizontalalignment('center')

    for x, y, c, h, w in zip(x, y, z, dx, dy):
        ax.add_artist(Rectangle(xy=(x, y),
                    color=cmap(c),
                    width=w, height=h))

    #plt.ylabel(product_components)

    plt.xlim([0, n])
    plt.ylim([0, n])
    plt.gca().invert_yaxis()
    plt.grid()
    st.pyplot(fig)



""" def random_DSM(order: int) -> list:
    '''
    Returns a random DSM of the specified order (dimension).

        Parameters:
            order (int):    The order (dimension) of the desired matrix.
        
        Returns:
            dsm (list):     A random DSM of dimension "order"
    '''
    dsm = []
    for i in range(order):
        row = []
        for j in range(order):
            if j == i:
                row.append(0)
            else:
                row.append(random.randint(0, 1))
        dsm.append(row)
    return dsm """


""" def random_prob_matrix(aDSM: list):
    '''
    Returns a random probability matrix for aDSM.

        Parameters:
            aDSM (list):                A DSM
        
        Returns:
            probability_matrix (list):  A random probability matrix
    '''
    probability_matrix = copy.deepcopy(aDSM)
    for i, row in enumerate(probability_matrix):
        for j, element in enumerate(row):
            if element == 1:
                probability_matrix[i][j] = random.random()
    return probability_matrix """


""" def plot_propagation_tree(propagation_tree):
        labels = nx.get_node_attributes(propagation_tree, 'name') 
        pos = hierarchy_pos(propagation_tree)
        plt.figure(figsize=(40,10))
        options = {"with_labels": True, 
                "node_shape": "s", 
                "arrows": False,
                "node_color": "white", 
                "edgecolors": "black", 
                "node_size": 1500}
        nx.draw_networkx(propagation_tree, 
                        pos=pos, 
                        labels=labels, 
                        **options)
"""


""" def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
        the root will be found and used
    - if the tree is directed and this is given, then 
        the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
        then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter) """


""" def plot_graph(aMatrix: list):
    '''Plot a graph described by a DSM matrix A'''
    # Transform to a Numpy matrix for generating networkx graph
    aNumpyMatrix = np.matrix(aMatrix)
    g = nx.from_numpy_array(aNumpyMatrix, create_using=nx.DiGraph)

    # layout
    #pos = nx.spring_layout(G, iterations=50)
    pos = nx.spring_layout(g)
    labels = {}
    for i in range(Anp[0].size):
        labels[i] = str(i)
    # rendering
    plt.figure(figsize=(10,10))
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos, node_size=1000, arrows=True)
    nx.draw_networkx_labels(g, pos, labels)
    plt.axis('off') """


""" def plot_heatmap(aMatrix: list):
    '''Plot aMatrix A as a heatmap'''
    color_palette = sns.color_palette("Blues", as_cmap=True)
    ax = sns.heatmap(aMatrix, linewidth=0.5, cmap=color_palette)
    plt.yticks(rotation=0)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    # set aspect of all axis
    ax.set_aspect('equal','box')
    plt.show() """

