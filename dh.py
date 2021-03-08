"""
Created on Mon Mar  1 22:42:04 2021

@author: Haotian Teng
"""
import numpy as np
from numpy.random import choice
from typing import List, Tuple, Union, Iterable
from scipy.cluster.hierarchy import to_tree
from scipy.cluster.hierarchy import ClusterNode
class HierarchicalSampling(object):
    def __init__(self,linkage:np.ndarray, n_samples:int, n_class:int):
        """
        Implmentation of Hierarchical Sampling for Active Learning:
        https://icml.cc/Conferences/2008/papers/324.pdf

        Parameters
        ----------
        linkage : np.ndarray
            A linkage matrix from scipy.cluster.hierarchy.linkage.
        n_samples : int
            Number of samples.
        n_class : int
            Number of classes.

        Returns
        -------
        An instance for hierarchical sampling method.

        """
        self.n_samples = n_samples
        self.n_class = n_class
        self.btree, self.node_list = self._construct_btree(linkage)
        self.lc = np.array([x.count for x in self.node_list],dtype = np.int)
        self.n_nodes = len(self.node_list)
        self.major_label = np.zeros(self.n_nodes,dtype = np.int)
        self.c = np.zeros((self.n_nodes,n_class)) 
        #the count of each class in points sampled from node.
        self.p = np.zeros((self.n_nodes,n_class))
        #the fraction of each class in points sampled from node.
        self.sampled = np.zeros(self.n_nodes)
        #the number of points sampled in the subtree rooted at each node.
        self.A = np.zeros((self.n_nodes,n_class),dtype = bool)
        #the admmisible score of node i and label l.
        self.p_LB = np.zeros((self.n_nodes,n_class))
        #the lower bound of the probability for node i and label l.
    def _construct_btree(self,linkage):
        btree, node_list = to_tree(linkage,rd = True)
        btree.parent = None
        for node in node_list:
            if node.left:
                node.left.parent = node 
            if node.right:
                node.right.parent = node
            node.sampled = 0
            if node.is_leaf():
                node.queryed = False
        return btree, node_list
    
    @property
    def leaves_count(self):
        return self.lc
    
    def get_leaves(self, node:Union[ClusterNode,int],leaves:List[ClusterNode] = None)-> List:
        """
        Get the list of leaves node under the subtree.

        Parameters
        ----------
        node : Union[ClusterNode,int]
            The root node of the subtree.

        Returns
        -------
        List
            A list contain the leaves nodes.

        """
        node = self.node_list[node] if not isinstance(node,ClusterNode) else node
        if not leaves:
            leaves = []
        if node.is_leaf():
            leaves.append(node)
            return leaves
        else:
            leaves = self.get_leaves(node.left,leaves)
            leaves = self.get_leaves(node.right,leaves)
            return leaves
            
    def update_empirical(self,
                          current:Union[ClusterNode,int], 
                          subroot:Union[ClusterNode,int],
                          label:int):
        """
        Update empricial count for a given label from current node u to a
        subtree root node v.

        Parameters
        ----------
        current : Union[ClusterNode,int]
            The leaf node whose label gut queryed.
        subroot : Union[ClusterNode,int]
            The root node of the subtree pruning.
        label : int
            The label of current node.

        Returns
        -------
        None.

        """
        current = self.node_list[current] if not isinstance(current,ClusterNode) else current
        subroot = self.node_list[subroot] if not isinstance(subroot,ClusterNode) else subroot
        if not current.queryed:
            current.queryed = True
        while current and current.id <= subroot.id:
            self.c[current.id][label]+=1
            self.sampled[current.id] +=1
            current = current.parent
        self.p = self.c/self.sampled[:,None]
    
    def update_admissible(self,beta:float = 2.0):
        """
        Update the admissible score and the upper and lower bound.

        Parameters
        ----------
        beta : float, optional
            The hyperparameter beta, larger the beta, higher chance to expand
            more subtree along the tree. The default is 2.

        Returns
        -------
        None.

        """
        delta = 1/self.sampled[:,None]+np.sqrt(self.p*(1-self.p)/self.sampled[:,None])
        p_LB = np.fmax(self.p-delta,0)
        p_UB = np.fmin(self.p+delta,1)
        for l in np.arange(self.n_class):
            po_UB = np.delete(p_UB,l,axis = 1)
            self.A[:,l] = np.all(p_LB[:,l][:,None]>beta*po_UB-beta+1,axis = 1)
            # self.A[:,l] = p_LB[:,l]>1/3 #For 2 classes case.
        e_tilde = 1-self.p
        e_tilde[~self.A] = 1
        self.e_tilde = e_tilde
        self.p_LB = p_LB
    
    def best_pruning_and_labeling(self,
                                  prunning:Iterable[Union[ClusterNode,int]],
                                  beta:float = 2.0
                                  )-> Tuple[np.ndarray,int]:
        """
        Update admissible A and find the best prunning and the label for the 
        give non-leaf node.

        Parameters
        ----------
        prunning : Iterable[Union[ClusterNode,int]]
            The list of root of the selected subtree, the current "prunning".

        Returns
        -------
        prunning_ : numpy.ndarray
            A array given the best prunning node(s) given the subroot.
        label : int
            The major label of given subroot.

        """
        e_tilde = self.e_tilde
        prunning = [x.id if isinstance(x,ClusterNode) else x for x in prunning]
        score = np.zeros(len(self.node_list))
        new_prunnings = []
        for idx, node in enumerate(self.node_list):
            if node.is_leaf():
                score[idx] = np.nanmin(e_tilde[idx])
            else:
                score_curr = np.nanmin(e_tilde[idx])
                if np.any(self.A[idx,:]):
                    left = node.left
                    right = node.right
                    score_desc = left.count/node.count * score[left.id]+\
                                 right.count/node.count * score[right.id]
                    score[idx] = np.minimum(score_desc,score_curr)
                else:
                    score_desc = np.inf
                    score[idx] = score_curr
            if idx in prunning:
                if node.is_leaf():
                    label = np.nanargmin(e_tilde[idx])
                    prunning_ = [idx]
                else:
                    label = np.nanargmin(e_tilde[idx]) if score_curr<score_desc else np.where(self.A[idx,:])[0][0]
                    prunning_ = [idx] if score_curr<score_desc else [node.left.id,node.right.id]
                for p in prunning_:
                    self.major_label[p] = label
                new_prunnings.extend(prunning_)
        return new_prunnings
    
    def assign_labels(self, 
                       current:Union[ClusterNode,int], 
                       subroot:Union[ClusterNode,int]):
        """
        Assign label to the current node according to the root of subtree.

        Parameters
        ----------
        current : Union[ClusterNode,int]
            The node assign labels begins at.
        subroot : Union[ClusterNode,int]
            The root of the subtree node.

        Returns
        -------
        None.

        """
        subroot = self.node_list[subroot] if not isinstance(subroot, ClusterNode) else subroot
        current = self.node_list[current] if not isinstance(current, ClusterNode) else current
        if current.is_leaf():
            self.major_label[current.id] = self.major_label[subroot.id]
        else:
            self.assign_labels(current.left,subroot)
            self.assign_labels(current.right,subroot)
            
    def active_sampling(self,prunning:List[int])->int:
        """
        Active sampling of query prunning.

        Parameters
        ----------
        prunning : List[int]
            The current prunning of the tree.

        Returns
        -------
        int
            The node of prunning need to explore.

        """
        p_LB = self.p_LB[prunning]
        w = self.leaves_count[prunning]
        L = self.major_label[prunning]
        p_LB = p_LB[np.arange(len(p_LB)),L]
        prob = w*(1-p_LB)
        if np.sum(prob) == 0:
            return choice(prunning)
        prob = prob/prob.sum()
        return choice(prunning,p = prob)
            
            