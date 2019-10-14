import numpy as np
from scipy.special import comb
import pandas as pd
try:
    import cupy as cp
except:
    pass
import functools
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
try:
   import cPickle as pickle
except:
   import pickle
import multiprocessing
import itertools


class latent_linear_model:

    def __init__(self, X, N, model, order=1, multi_channel=False, compute_partition_function=True, D_eps=0, device='cpu', n_proc=1):
        """
        Parameters:
        -----------
        X: numpy.ndarray
            M x l1 x l2 x (colour channel):
        N: int
            Size of the dictionary. Ideally N > L.
        model: string
            'dl': Dictionary Learning
            'ica': Independent Component Analysis
        order: int
            Order of the higher-order feature interaction
        multi_channel: bool
            If the image contains multiple colour channels
        compute_partition_function: bool
            The model computes the partition function to
            ensure the data layer is normalized.
        D_eps: float
            Asign a probability to the dictionary layer.
            It is recommended that this value should be 0.
        device: string
            'cpu': Run critical step on cpu
            'gpu': Run critical step on gpu

        n_proc: int
            Number of processes. I have not done any proper
            testing for more than 1 processor. Results may
            be incorrect for more than 1 processor.

        Returns:
        --------
        None
        """
        # Initalize parameters
        self.N = N
        self.model = model.lower()
        self.order = order
        self.multi_channel = multi_channel
        self.compute_partition_function = compute_partition_function
        self.D_eps = D_eps
        self.device = device
        self.n_proc = n_proc
        self.psi = 0

        # Pre-process X
        self.X = np.array(X)
        if self.multi_channel == True:
            self.IN, self.IK, self.IL, self.IC = self.X.shape
        else:
            self.IN, self.IK, self.IL = self.X.shape
        self._normalize_input()
        if self.model == 'dl':
            self.X = self._flatten_X_dictionary_learning()
        elif self.model == 'ica':
            self.X = self._flatten_X_ICA()
        else:
            raise('Please selection model to be dictionary learning (DL) or indepdent component analysis (ICA)')

        # Create structure
        self._create_all_layers()
        self._compute_eta_emp()

        # Perform single step to update parameters
        self._update_model_parameters()

        # Initalize parameters for matching pursuit
        self.l = np.full(self.X.shape[1], N)
        self.L = self.R_theta_mat.shape[0]

        # Initalize arrays to store values
        self.current_iterations = 0
        self.stored_iteration_list = list()
        self.stored_RMSE_list = list()
        self.stored_runtime = list()

    def _normalize_input(self):
        """
        Normalizes X so that the sum of all elements
        in the input is equal to 1. The normalization
        will be stored as a class variable.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self.Z = np.sum(self.X)
        self.X /= self.Z

    def _flatten_X_dictionary_learning(self):
        """
        Flattens a 2D image to a single vector.
        Each column represents an image.
        Each row represents a pixel.

        Parameters:
        -----------
        None

        Returns:
        --------
        X: numpy.ndarray
            Flattened image
        """
        X = self.X.reshape(self.IN, self.IK * self.IL).T
        return X

    def unflatten_X_dictionary_learning(self, X):
        """
        Convert the flattened image to a 2D matrix.

        Parameters:
        -----------
        X: numpy.ndarray
            Unflattened matrix with dimensions
            (number of image x pixel row x pixel column)

        Returns:
        --------
        X: numpy.ndarray

        """
        X = X.T.reshape(self.IN, self.IK, self.IL)
        return X

    def _flatten_X_ICA(self):
        """
        Flattens a 2D image to a single vector.

        Paramters:
        ----------
        None

        Returns:
        --------
        A matrix where the rows are each pixel
        and each column is a image.        
        """
        if self.multi_channel:
            X = self.X.reshape(self.IN, self.IK * self.IL * self.IC)
        else:
            X = self.X.reshape(self.IN, self.IK * self.IL)
        return X

    def unflatten_X_ICA(self, X):
        """
        Convert the flattened image to a 2D matrix

        Parameters:
        -----------
        X: numpy.ndarray
            A matrix where the rows are each pixel
            and each column is a image.

        Returns:
        --------
        X: numpy.ndarray
            A tensor (Number of Image x pixel row x pixel column x channel)
        """
        if self.multi_channel:
            X = X.reshape(self.IN, self.IK, self.IL, self.IC)
        else:
            X = X.reshape(self.IN, self.IK, self.IL)
        return X

    def unflatten_R_ICA(self, C):
        """
        Convert the flattened image to a 2D matrix

        Parameters:
        -----------
        C: numpy.ndarray
            The flattened matrix where the matrix dimension
            is (pixel x images)

        Returns:
        --------
        C: numpy.ndarray
            A tensor with the following dimensions
            (images x pixel row x pixel column x channel)
        """
        if self.multi_channel:
            C = C.reshape(self.N, self.IK, self.IL, self.IC)
        else:
            C = C.reshape(self.N, self.IK, self.IL)
        return C

    def _create_data_layer(self):
        """
        Create parameters for the data layer
        The size of the data layer is MxL

        Paramters:
        ----------
        None

        Returns:
        --------
        None
        """
        shape = self.X.shape
        self.X_p_mat = np.ones(shape)
        self.X_eta_mat = np.zeros(shape)

    def _create_representation_layer(self):
        """
        Create parameters for the representation layer
        The size of the representation layer is NxM

        Paramters:
        ----------
        None

        Returns:
        --------
        None
        """
        shape = (self.N, self.X.shape[1])
        self.R_theta_mat = np.zeros(shape)
        self.R_p_mat = np.ones(shape)
        self.R_eta_mat = np.zeros(shape)

    def _create_dictionary_layer(self):
        """
        Create parameters for the dictionary layer
        The size of the dictionary layer is LxN

        Paramters:
        ----------
        None

        Returns:
        --------
        None
        """
        shape = (self.X.shape[0], int(np.sum(comb(self.N,np.arange(self.order)+1))))
        self.D_theta_mat = np.zeros(shape)
        self.D_p_mat = np.ones(shape)
        self.D_eta_mat = np.zeros(shape)

    def _compute_partition_function(self):
        """
        Compute partition function.
        Note: It is added to the previous partition
        function because the previous parition function
        has been included in the calculations of the
        probabilities.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        if self.compute_partition_function:
            psi = self.psi + np.log(np.sum(self.X_p_mat)
                + np.sum(self.R_p_mat) + np.sum(self.D_p_mat))
        else:
            psi = self.psi
        return psi

    def _activate_nodes(self, C_vec, C, ci, N, depth):
        """
        A recursive function to generate only the nodes which
        are required.

        Parameters:
        -----------
        C_vec: numpy.ndarray
            Matrix used to store the output
        C: numpy.ndarray
            One possible combination
        ci: int
            Current index
        N: int
            Size of the array
        depth: int
            Current depth of the recussion

        Returns:
        --------
        None
        """
        if depth == 0:
            C_vec.append(C)
            return

        for c in range(ci, N-depth+1):
            C_current = np.array(C, dtype=bool)
            C_current[c] = True
            C_current = self._activate_nodes(C_vec, C_current, c+1, N, depth-1)

    def _get_higher_order_nodes(self, N, order):
        """
        Returns a array of arrays, where each row
        represents an active node for the order
        of interaction specified. N represents the
        number of elements in the node.

        Parameters:
        -----------
        N: int
            Number of nodes (or elements)
        order: int
            Order of interaction

        Returns:
        --------
        C_vec: numpy.ndarray
            Each row is one of all possible combinations
            for the higher order interaction
        """
        C_vec = list()
        for d in np.arange(order)+1:
            self._activate_nodes(C_vec, np.zeros(N, dtype=bool), 0, N, d)
        C_vec = np.array(C_vec, dtype=bool)
        return C_vec

    def _R_eta_worker(self, ri, rj):
        out = functools.reduce(lambda x, y: y.union(x) , self.X_eta_set_mat[:,rj], self.R_set_mat[ri,rj])
        return ri, rj, out

    def _D_eta_worker(self, di, dj):
        out = functools.reduce(lambda x, y: y.union(x) , self.R_eta_set_mat[self.higher_order_index[dj,:],:].flatten(), self.D_set_mat[di,dj])
        return di, dj, out

    def _R_theta_worker(self, ri, rj):
        out = functools.reduce(lambda x, y: y.union(x) , self.D_theta_set_mat[:,self.higher_order_index[:,ri]].flatten(), self.R_set_mat[ri,rj])
        return ri, rj, out

    def _X_theta_worker(self, xi, xj):
        out = functools.reduce(lambda x, y: y.union(x) , self.R_theta_set_mat[:,xj], self.X_set_mat[xi,xj])
        return xi, xj, out

    def _create_sets(self):
        """
        Create an array of sets for eta (upper set)
        and theta (lower set). The output is stored
        in the class variables.

        Paramters:
        ----------
        None

        Returns:
        --------
        None
        """
        # Create set for data layer
        self.X_set_mat = np.array([{'x-{}-{}'.format(xi, xj)} for xi in range(self.X_eta_mat.shape[0]) for xj in range(self.X_eta_mat.shape[1])]).reshape(self.X_eta_mat.shape)

        # Create set for representation layer
        self.R_set_mat = np.array([{'r-{}-{}'.format(ri, rj)} for ri in range(self.R_eta_mat.shape[0]) for rj in range(self.R_eta_mat.shape[1])]).reshape(self.R_eta_mat.shape)

        # Create set for data layer
        self.D_set_mat = np.array([{'d-{}-{}'.format(di, dj)} for di in range(self.D_eta_mat.shape[0]) for dj in range(self.D_eta_mat.shape[1])]).reshape(self.D_eta_mat.shape)

        # Create eta set in the data layer
        self.X_eta_set_mat = np.array(self.X_set_mat)

        # Create eta set for the representation layer
        self.R_eta_set_mat = np.full(self.R_set_mat.shape, set())
        if self.n_proc == 1:
            for ri in range(self.R_eta_set_mat.shape[0]):
                for rj in range(self.R_eta_set_mat.shape[1]):
                    self.R_eta_set_mat[ri,rj] = functools.reduce(lambda x, y: y.union(x) , self.X_eta_set_mat[:,rj], self.R_set_mat[ri,rj])
        else:
            combinations = itertools.product(range(self.R_eta_set_mat.shape[0]), range(self.R_eta_set_mat.shape[1]))
            pool = multiprocessing.Pool(processes=self.n_proc)
            out = pool.starmap(self._R_eta_worker, combinations)
            pool.close()
            pool.join()
            for o in out:
                self.R_eta_set_mat[o[0], o[1]] = o[2]

        # Create eta set for the dictionary layer
        self.D_eta_set_mat = np.full(self.D_set_mat.shape, set())
        if self.n_proc == 1:
            for di in range(self.D_eta_set_mat.shape[0]):
                for dj in range(self.D_eta_set_mat.shape[1]):
                    self.D_eta_set_mat[di,dj] = functools.reduce(lambda x, y: y.union(x) , self.R_eta_set_mat[self.higher_order_index[dj,:],:].flatten(), self.D_set_mat[di,dj])
        else:
            combinations = itertools.product(range(self.D_eta_set_mat.shape[0]), range(self.D_eta_set_mat.shape[1]))
            pool = multiprocessing.Pool(processes=self.n_proc)
            out = pool.starmap(self._D_eta_worker, combinations)
            pool.close()
            pool.join()
            for o in out:
                self.D_eta_set_mat[o[0], o[1]] = o[2]

        # Create theta set in the dictionary layer
        self.D_theta_set_mat = np.array(self.D_set_mat)

        # Create theta set in the representation layer
        self.R_theta_set_mat = np.full(self.R_set_mat.shape, set())
        if self.n_proc == 1:
            for ri in range(self.R_eta_set_mat.shape[0]):
                for rj in range(self.R_eta_set_mat.shape[1]):
                    self.R_theta_set_mat[ri,rj] = functools.reduce(lambda x, y: y.union(x) , self.D_theta_set_mat[:,self.higher_order_index[:,ri]].flatten(), self.R_set_mat[ri,rj])
        else:
            combinations = itertools.product(range(self.R_theta_set_mat.shape[0]), range(self.R_theta_set_mat.shape[1]))
            pool = multiprocessing.Pool(processes=self.n_proc)
            out = pool.starmap(self._R_theta_worker, combinations)
            pool.close()
            pool.join()
            for o in out:
                self.R_theta_set_mat[o[0], o[1]] = o[2]

        # Create theta set in the data layer
        self.X_theta_set_mat = np.full(self.X_set_mat.shape, set())
        if self.n_proc == 1:
            for xi in range(self.X_theta_set_mat.shape[0]):
                for xj in range(self.X_theta_set_mat.shape[1]):
                    self.X_theta_set_mat[xi,xj] = functools.reduce(lambda x, y: y.union(x) , self.R_theta_set_mat[:,xj], self.X_set_mat[xi,xj])
        else:
            combinations = itertools.product(range(self.X_theta_set_mat.shape[0]), range(self.X_theta_set_mat.shape[1]))
            pool = multiprocessing.Pool(processes=self.n_proc)
            out = pool.starmap(self._X_theta_worker, combinations)
            pool.close()
            pool.join()
            for o in out:
                self.X_theta_set_mat[o[0], o[1]] = o[2]

    def _create_all_layers(self):
        """
        Create all layers and initalise values.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self._create_data_layer()
        self._create_representation_layer()
        self._create_dictionary_layer()
        self.psi = self._compute_partition_function()
        self.higher_order_index = self._get_higher_order_nodes(self.N, self.order)
        self._create_sets()
        # self._remove_connection_from_dictionary()
        self._remove_connection_from_representation()

    def _remove_connection_from_dictionary(self):
        counter = 0            
        for di in range(self.D_eta_set_mat.shape[0]):
            for dj in range(self.D_eta_set_mat.shape[1]):
                n = counter % self.R_set_mat.shape[1]
                for k in np.where(self.higher_order_index[dj,:])[0]:
                    self.D_eta_set_mat[di,dj] = self.D_eta_set_mat[di,dj].difference(self.R_eta_set_mat[k, n])
                    self.R_theta_set_mat[k, n] = self.R_theta_set_mat[k, n].difference(self.D_theta_set_mat[di, dj])
                counter += 1

    def _R_remove_connection_worker(self, inp):
        n = inp[0] % self.X_set_mat.shape[0]
        rj = inp[1][0]
        ri = inp[1][1]
        out1 = self.R_eta_set_mat[ri,rj].difference(self.X_eta_set_mat[n, rj])
        out2 = self.X_theta_set_mat[n, rj].difference(self.R_theta_set_mat[ri, rj])

        return ri, rj, n, out1, out2

    def _remove_connection_from_representation(self):
        """
        Systematically remove a single edge from each node
        in the representation layer.

        Parameters:
        ----------
        None

        Returns:
        --------
        None
        """
        if self.n_proc == 1:
            counter = 0
            for rj in range(self.R_eta_set_mat.shape[1]):
                for ri in range(self.R_eta_set_mat.shape[0]):
                    n = counter % self.X_set_mat.shape[0]
                    self.R_eta_set_mat[ri,rj] = self.R_eta_set_mat[ri,rj].difference(self.X_eta_set_mat[n, rj])
                    self.X_theta_set_mat[n, rj] = self.X_theta_set_mat[n, rj].difference(self.R_theta_set_mat[ri, rj])
                    counter += 1
        else:
            combinations = itertools.product(range(self.R_eta_set_mat.shape[1]), range(self.R_eta_set_mat.shape[0]))
            pool = multiprocessing.Pool(processes=self.n_proc)
            out = pool.map(self._R_remove_connection_worker, enumerate(combinations))
            pool.close()
            pool.join()
            for o in out:
                self.R_eta_set_mat[o[0], o[1]] = o[3]
                self.X_theta_set_mat[o[2], o[1]] = o[4]

    def _compute_p_d(self, psi):
        """
        Compute probability in the dictionary layer

        Parameters:
        ----------
        psi: float
            The partition function

        Returns:
        --------
        D_p_mat: numpy.ndarray
            Returns the probability matrix for the
            dictionary layer.
        """
        D_p_mat = np.exp(self.D_theta_mat - psi)
        return D_p_mat

    def _compute_p_r(self, psi):
        """
        Compute probability in the representation layer

        Parameters:
        ----------
        psi: float
            The partition function

        Returns:
        --------
        R_p_mat: numpy.ndarray
            Returns the probability matrix for the
            representation layer.
        """
        R_p_mat = np.exp(self._compute_sum_of_set_mat(self.R_theta_set_mat, param='theta', domain='model') - psi)
        return R_p_mat

    def _compute_p_x(self, psi):
        """
        Compute probability in the data layer

        Parameters:
        ----------
        psi: float
            The partition function

        Returns:
        --------
        X_p_mat: numpy.ndarray
            Returns the probability matrix for the
            data layer.
        """
        X_p_mat = np.exp(self._compute_sum_of_set_mat(self.X_theta_set_mat, param='theta', domain='model') - psi)
        return X_p_mat

    def _compute_all_p(self):
        """
        Compute probability in all layers

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self.D_p_mat = self._compute_p_d(self.psi)
        self.R_p_mat = self._compute_p_r(self.psi)
        self.X_p_mat = self._compute_p_x(self.psi)

    def _compute_eta_x(self):
        """
        Compute eta in the data layer

        Parameters:
        -----------
        None

        Returns:
        --------
        X_eta_mat: numpy.ndarray
            Returns the eta matrix for the
            data layer.
        """
        X_eta_mat = np.array(self.X_p_mat)
        return X_eta_mat

    def _compute_eta_r(self):
        """
        Compute eta in the representation layer

        Parameters:
        ----------
        psi: float
            The partition function

        Returns:
        --------
        R_eta_mat: numpy.ndarray
            Returns the eta matrix for the
            representation layer.
        """
        R_eta_mat = self._compute_sum_of_set_mat(self.R_eta_set_mat, param='p', domain='model')
        return R_eta_mat

    def _compute_eta_d(self):
        """
        Compute eta in the dictionary layer

        Parameters:
        ----------
        psi: float
            The partition function

        Returns:
        --------
        D_eta_mat: numpy.ndarray
            Returns the eta matrix for the
            dictionary layer.
        """
        D_eta_mat = self._compute_sum_of_set_mat(self.D_eta_set_mat, param='p', domain='model')
        return D_eta_mat

    def _compute_all_eta(self):
        """
        Compute eta in all layers

        Paramters:
        ----------
        None

        Returns:
        --------
        None
        """
        self.X_eta_mat = self._compute_eta_x()
        self.R_eta_mat = self._compute_eta_r()
        self.D_eta_mat = self._compute_eta_d()

    def _compute_eta_emp(self):
        """
        Compute eta in the emperical dataset

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self.R_eta_emp_mat = self._compute_sum_of_set_mat(self.R_eta_set_mat, domain='emperical')
        self.D_eta_emp_mat = self._compute_sum_of_set_mat(self.D_eta_set_mat, domain='emperical')
        self.D_eta_emp_mat -= self.D_eps * np.random.rand(*self.D_eta_emp_mat.shape)

    # def _decorrelation(self, v):
    #     """
    #     Apply a symmetric decorrelation to the matrix so that each atom
    #     in the dictionary is decorrelated.
    #     """
    #     gram = np.dot(v.T, v)
    #     s, u = np.linalg.eigh(gram + (eps * np.eye(gram.shape[0]))) # Small value added on the diagonal to avoid negatives numbers from floating point errors
    #     v_decorrelated = np.dot(v, np.dot(u * (1 / np.sqrt(s)), u.T))
    #     return v_decorrelated

    def _compute_sum_of_set(self, input_set, param=None, domain='model'):
        """
        Give returns sum of a parameter.

        Paramters:
        ----------
        input_set: set
            A set containing strings with param-index1-index2
        param: string
            'p': probability
            'theta': theta
        domain: string
            'model': For model distribution
            'emperical': For emperical distribution

        Returns:
        v: float
            Sum of a parameter
        """
        if not (domain == 'model' or domain == 'emperical'):
            raise("domain should be either 'model' or 'emperical'")

        # Select parameters
        if param == 'p':
            X_mat = self.X_p_mat
            R_mat = self.R_p_mat
            D_mat = self.D_p_mat
        elif param == 'theta':
            X_mat = np.zeros(self.X_p_mat.shape)
            R_mat = self.R_theta_mat
            D_mat = self.D_theta_mat

        input_set = set(input_set)
        v = 0

        if domain == 'model':
            while input_set:
                s = input_set.pop()
                s = s.split('-')
                if s[0] == 'x':
                    v += X_mat[int(s[1]), int(s[2])]
                elif s[0] == 'r':
                    v += R_mat[int(s[1]), int(s[2])]
                elif s[0] == 'd':
                    v += D_mat[int(s[1]), int(s[2])]

        if domain == 'emperical':
            while input_set:
                s = input_set.pop()
                s = s.split('-')
                if s[0] == 'x':
                    v += self.X[int(s[1]), int(s[2])]
        return v

    def _compute_sum_of_set_mat_worker(self, i, j, S, param, domain):
        out = self._compute_sum_of_set(S[i,j], param=param, domain=domain)
        return i, j, out

    def _compute_sum_of_set_mat(self, S, param=None, domain='model'):
        """
        For a given set. It will return the same of
        the given parameter.

        Parameters:
        -----------
        S: numpy.array
            An matrix of input sets.
        param: string
            'p': Compute parameters for p.
            'theta': Compute paramters for theta.
        domain: string
            'model': Computes sum of sets for the model distribution.
            'emperical': Compute sum of sets for the emperical distribution.

        Returns:
        --------
        param_sum: numpy.ndarray
            Returns a matrix of the sum of parameters
        """
        param_sum = np.zeros(S.shape)
        # if self.n_proc == 1:
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                param_sum[i,j] = self._compute_sum_of_set(S[i,j], param=param, domain=domain)
        # else:
        #     combinations = itertools.product(range(S.shape[0]), range(S.shape[1]), [S], [param], [domain])
        #     pool = multiprocessing.Pool(processes=self.n_proc)
        #     out = pool.starmap_async(self._compute_sum_of_set_mat_worker, combinations)
        #     pool.close()
        #     pool.join()
        #     for o in out:
        #         param_sum[o[0], o[1]] = o[2]

        return param_sum

    def _fisher_information_worker(self, i, j, eta_set_flat, eta_flat):
        intersection_set = eta_set_flat[i].intersection(eta_set_flat[j])
        out = self._compute_sum_of_set(intersection_set, param='p', domain='model') - (eta_flat[i] * eta_flat[j])
        return i, j, out

    def _D_fisher_information_matrix(self):
        """
        Returns the fisher information matrix for the
        dictionary layer

        Parameters:
        -----------
        None

        Returns:
        --------
        D_fisher_information_matrix: numpy.ndarray
            Fisher information matrix for the dictionary layer
        """
        D_size = np.prod(self.D_eta_mat.shape)

        D_fisher_information_matrix = np.zeros((D_size, D_size))
        D_eta_flat = self.D_eta_mat.flatten()

        D_eta_set_flat = self.D_eta_set_mat.flatten()

        # if self.n_proc == 1:
        for di in np.arange(D_size):
            for dj in np.arange(di, D_size):
                intersection_set = D_eta_set_flat[di].intersection(D_eta_set_flat[dj])
                g = self._compute_sum_of_set(intersection_set, param='p', domain='model') - (D_eta_flat[di] * D_eta_flat[dj])
                D_fisher_information_matrix[di, dj] = g
                D_fisher_information_matrix[dj, di] = g
        # else:
        #     combinations = np.array([(di, dj, D_eta_set_flat, D_eta_flat) for di in np.arange(D_size) for dj in np.arange(di, D_size)])
        #     pool = multiprocessing.Pool(processes=self.n_proc)
        #     out = pool.starmap(self._fisher_information_worker, combinations)
        #     pool.close()
        #     pool.join()
        #     for o in out:
        #         D_fisher_information_matrix[o[0], o[1]] = o[2]
        #         D_fisher_information_matrix[o[1], o[0]] = o[2]
        
        return D_fisher_information_matrix

    def _R_fisher_information_matrix(self):
        """
        Returns the fisher information matrix for the
        representation layer

        Parameters:
        -----------
        None

        Returns:
        --------
        R_fisher_information_matrix: numpy.ndarray
            Fisher information matrix for the representation layer
        """
        R_size = np.prod(self.R_eta_mat.shape)

        R_fisher_information_matrix = np.zeros((R_size, R_size))
        R_eta_flat = self.R_eta_mat.flatten()

        R_eta_set_flat = self.R_eta_set_mat.flatten()

        # if self.n_proc == 1:
        for ri in np.arange(R_size):
            for rj in np.arange(ri, R_size):
                intersection_set = R_eta_set_flat[ri].intersection(R_eta_set_flat[rj])
                g = self._compute_sum_of_set(intersection_set, param='p', domain='model') - (R_eta_flat[ri] * R_eta_flat[rj])
                R_fisher_information_matrix[ri, rj] = g
                R_fisher_information_matrix[rj, ri] = g
        # else:
        #     combinations = np.array([(ri, rj, R_eta_set_flat, R_eta_flat) for ri in np.arange(R_size) for rj in np.arange(ri, R_size)])
        #     pool = multiprocessing.Pool(processes=self.n_proc)
        #     out = pool.starmap(self._fisher_information_worker, combinations)
        #     pool.close()
        #     pool.join()
        #     for o in out:
        #         R_fisher_information_matrix[o[0], o[1]] = o[2]
        #         R_fisher_information_matrix[o[1], o[0]] = o[2]

        return R_fisher_information_matrix

    def _step(self, lr, natural_gradient=False):
        """
        Single step for gradient descent.

        Parameters:
        -----------
        lr: float
            Learning rate
        natural_gradient: bool
            Use natural gradient to solve the problem (recommended)
            However for larger problems there may be numerical problems
            since the numbers cannot be stored in float64.

        Returns:
        --------
        error: float
            Root mean square error
        """
        # Compute gradient
        if self.device == 'cpu':
            D_gradient = self.D_eta_mat - self.D_eta_emp_mat
            R_gradient = self.R_eta_mat - self.R_eta_emp_mat
        elif self.device == 'gpu':
            D_gradient = cp.array(self.D_eta_mat) - cp.array(self.D_eta_emp_mat)
            R_gradient = cp.array(self.R_eta_mat) - cp.array(self.R_eta_emp_mat)

        if natural_gradient:
            D_fim = self._D_fisher_information_matrix()
            R_fim = self._R_fisher_information_matrix()

            D_gradient_flat = D_gradient.flatten()[:,None]
            R_gradient_flat = R_gradient.flatten()[:,None]

            if self.device == 'cpu':
                D_natural_gradient = np.linalg.solve(D_fim, D_gradient_flat)
                R_natural_gradient = np.linalg.solve(R_fim, R_gradient_flat)
            elif self.device == 'gpu':
                D_fim = cp.array(D_fim)
                R_fim = cp.array(R_fim)
                D_natural_gradient = cp.linalg.solve(D_fim, D_gradient_flat)
                R_natural_gradient = cp.linalg.solve(R_fim, R_gradient_flat)

            D_gradient = D_natural_gradient.reshape(self.D_eta_mat.shape)
            R_gradient = R_natural_gradient.reshape(self.R_eta_mat.shape)

        # Apply gradient descent
        if self.device == 'cpu':
            self.D_theta_mat -= lr * D_gradient
            self.R_theta_mat -= lr * R_gradient
        elif self.device == 'gpu':
            self.D_theta_mat = cp.asnumpy(cp.array(self.D_theta_mat) - lr * D_gradient)
            self.R_theta_mat = cp.asnumpy(cp.array(self.R_theta_mat) - lr * R_gradient)



        # Decorrelate dictionary
        # self.D_theta_mat = self._decorrelation(self.D_theta_mat)
        self._update_model_parameters()

    def _update_model_parameters(self):
            """
            Update all model parameters.
            updates parameters
            Updates all p and eta in dictionary, representation and data layer

            Paramters:
            ----------
            None

            Returns:
            --------
            None
            """
            self._compute_all_p()
            self.psi = self._compute_partition_function()
            self._compute_all_p()
            self._compute_all_eta()

    # def _matching_pusuit_step(self, n, m):
    #     """

    #     """
    #     # Remove node
    #     self.R_theta_mat.mask[n,m] = True
    #     self._update_model_parameters()

    #     # Compute error
    #     error = np.sum((self.X - self.X_p_mat)**2)**0.5

    #     # Restore original conifuration
    #     self.R_theta_mat.mask[n,m] = False
    #     self._update_model_parameters()

    #     return error

    # def _matching_pusuit(self):
    #     """

    #     """
    #     error_mat = np.full(self.R_theta_mat.shape, np.inf)
    #     for m in range(self.R_theta_mat.shape[1]):
    #         for n in range(self.R_theta_mat.shape[0]):
    #             if self.R_theta_mat.mask[n,m] == True:
    #                 continue
    #             error_mat[n,m] = self._matching_pusuit_step(n,m)
    #     print('error_mat:\n', error_mat)
    #     print('np.argmin(error_mat, axis=0):\n', np.argmin(error_mat, axis=0))
    #     while True:
    #         min_error_idx = np.unravel_index(np.argmin(error_mat), shape=error_mat.shape)
    #         if self.l[min_error_idx[1]] > self.L:
    #             break
    #         else:
    #             error_mat[min_error_idx] = np.inf
    #     self.R_theta_mat.mask[min_error_idx] = True

    #     self.l[min_error_idx[1]] -= 1
    #     self._update_model_parameters()
    #     print('R_theta_mat.mask:\n', self.R_theta_mat.mask)
    #     print('R_theta_mat:\n', self.R_theta_mat)

    def _train(self, lr=0.01, iterations=10, natural_gradient=False, tol=1e-7, sparse_reprensentation=False, L=None, verbose=False, verbose_step=100, store_values=False, save_at_tol=None, save_path='./'):
        """
        Parameters:
        -----------
        lr: float
            Learning rate
        iterations: int
            Number of iterations
        natural_gradient: bool
            Use natural gradient to solve the problem (recommended)
            However for larger problems there may be numerical problems
            since the numbers cannot be stored in float64.
        tol: float
            Tolerence to converge
        sparse_reprensentation: bool
            Not implemented yet, do not use
        L: int
            Not implemented yet, do not use
        verbose: bool
            Print status
        verbose_step: int
            Number of iterations before printing status

        Returns:
        --------
        None
        """
        save_at_tol = np.sort(np.atleast_1d(save_at_tol))[::-1]
        save_idx = 0
        for i in np.arange(self.current_iterations, iterations)+1:
            current_iterations = i
            # Store previous theta to compute error
            if verbose and (i % verbose_step == 0):
                prev_R_theta_mat = np.array(self.R_theta_mat)
                prev_D_theta_mat = np.array(self.D_theta_mat)

            # Step model
            t0 = time.time()
            self._step(lr=lr, natural_gradient=natural_gradient)
            t1 = time.time()
            run_time = t1-t0

            RMSE_reconstruction_error = np.mean(((self.X - (self.X_p_mat / np.sum(self.X_p_mat)))*self.Z)**2)**0.5
            RMSE_gradient_error = ((np.sum((self.R_eta_mat - self.R_eta_emp_mat)**2) + np.sum((self.D_eta_mat - self.D_eta_emp_mat)**2)) / (np.prod(self.D_eta_mat.shape) + np.prod(self.R_eta_mat.shape) ))**0.5
            if self.model == 'dl':
                error = RMSE_reconstruction_error
            if self.model == 'ica':
                error = RMSE_gradient_error

            if store_values:
                self._store_values(RMSE_reconstruction_error, run_time)

            if (not (save_at_tol is None)) and (save_idx < len(save_at_tol)):
                while (save_idx < len(save_at_tol)):
                    if (save_at_tol[save_idx] > error):
                        tol_save_path = '{}/tol{:.0e}/'.format(save_path, save_at_tol[save_idx])
                        self.save_result(tol_save_path)
                        save_idx += 1
                    else:
                        break

            # Print status
            if verbose and (i % verbose_step == 0):
                print('======================================')
                print('iterations:', i, 'error:', RMSE_gradient_error)
                print('iterations:', i, 'reconstruction MAE error:', np.mean(np.abs((self.X - (self.X_p_mat / np.sum(self.X_p_mat)))*self.Z)))
                print('iterations:', i, 'reconstruction RMSE error:', RMSE_reconstruction_error)
                print('iterations:', i, 'theta change:', (np.sum((self.R_theta_mat - prev_R_theta_mat)**2) + np.sum((self.D_theta_mat - prev_D_theta_mat)**2) / (np.prod(self.D_eta_mat.shape) + np.prod(self.R_eta_mat.shape) )**0.5))
                print('iterations:', i, 'run time:', run_time)
                print('iterations:', i, 'D_theta_mat:', self.D_theta_mat)
                print('======================================')

            if tol > error:
                break

    def fit(self, lr=0.01, iterations=10, natural_gradient=False, tol=1e-7, sparse_reprensentation=False, L=None, verbose=False, verbose_step=100, store_values=False, save_at_tol=None, save_path='./'):
        """
        Parameters:
        -----------
        lr: float
            Learning rate
        iterations: int
            Number of iterations
        natural_gradient: bool
            Use natural gradient to solve the problem (recommended)
            However for larger problems there may be numerical problems
            since the numbers cannot be stored in float64.
        tol: float
            Tolerence to converge
        sparse_reprensentation: bool
            Not implemented yet, do not use
        L: int
            Not implemented yet, do not use
        verbose: bool
            Print status
        verbose_step: int
            Number of iterations before printing status

        Returns:
        --------
        None
        """
        # L = self.R_theta_mat.shape[0] if L is None else L
        self._train(lr=lr, iterations=iterations, natural_gradient=natural_gradient, tol=tol, sparse_reprensentation=sparse_reprensentation, L=L, verbose=verbose, verbose_step=verbose_step, store_values=store_values, save_at_tol=save_at_tol, save_path=save_path)

    def reconstruct_data(self, domain='model'):
        """
        Use the model to reconstruct the original input

        Parameters:
        -----------
        domain: string
            'model': Returns model distribution
            'emperical': Returns emperical distribution

        Returns:
        --------
        X: numpy.ndarray
            Reconstructed data
        """
        # Make copy of X_p
        if domain == 'model':
            X = np.array(self.X_p_mat)
        elif domain == 'emperical':
            X = np.array(self.X)
        
        if self.model == 'dl':
            X = self.unflatten_X_dictionary_learning(X)
        elif self.model == 'ica':
            X = self.unflatten_X_ICA(X)
        X *= self.Z
        return X

    def reconstruct_components(self):
        """
        Reconstruct the latent components

        Parameters:
        ----------
        None

        Returns:
        --------
        C: numpy.ndarray
            Returns matrix of latent components
            Pixels x images
        """
        C = self._compute_p_r(0)
        C = self.unflatten_R_ICA(C)
        C /= np.max(C)
        return C

    def _store_values(self, RMSE_reconstruction_error, run_time):
        """
        This function is the store the iterations
        root mean squared error of the reconstruction
        and the runtime into an array.

        Parameters:
        -----------
        RMSE_reconstruction_error: float
            Root mean square of the reconstruction error
        run_time: float
            Time it took for this iteration

        Returns:
        --------
        None
        """
        self.stored_iteration_list.append(self.current_iterations)
        self.stored_RMSE_list.append(RMSE_reconstruction_error)
        self.stored_runtime.append(run_time)

    def _write_results(self, path):
        """
        Writes out the stored information to a given path.

        Parameters:
        -----------
        path: string
            Path to save results.

        Returns:
        --------
        None
        """
        if not os.path.exists(path):
            os.makedirs(path)

        data = np.vstack([self.stored_iteration_list, self.stored_RMSE_list, self.stored_runtime]).T
        df = pd.DataFrame(data, columns=['Iterations', 'Reconstruction_RMSE', 'Runtime'])
        df['Runtime_cumsum'] = df['Runtime'].cumsum()
        df.to_csv('{}/model_data_per_iteration.csv'.format(path), index=False)

    def preprocess_output(self, X):
        X = np.array(X)
        if np.min(X) < 0:
            X -= np.min(X)
        X = np.array((X / np.max(X)) * 254, dtype=np.uint8)
        return X

    def _save_input(self, path):
        """
        Saves the normalised input data

        Parameters:
        -----------
        path: string
            Path to save results.

        Returns:
        --------
        None
        """
        X = self.reconstruct_data(domain='emperical')
        for i, x in enumerate(X):
            sub_path = './{}/Input/'.format(path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
            Image.fromarray(self.preprocess_output(x)).save('{}/Input/Input_{}.png'.format(path, i), format='png')

    def _save_model_reconstruction(self, path):
        """
        Parameters:
        -----------
        path: string
            Path to save results.

        Returns:
        --------
        None
        """
        X = self.reconstruct_data(domain='model')
        for i, x in enumerate(X):
            sub_path = '{}/Data_Reconstructed/'.format(path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
            Image.fromarray(self.preprocess_output(x)).save('{}/Data_Reconstructed/Data_Reconstructed_{}.png'.format(path, i), format='png')

    def _save_representation(self, path):
        """
        Parameters:
        -----------
        path: string
            Path to save results.

        Returns:
        --------
        None
        """
        C = self.reconstruct_components()
        for i, c in enumerate(C):
            sub_path = '{}/Components_Reconstructed/'.format(path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
            Image.fromarray(self.preprocess_output(c)).save('{}/Components_Reconstructed/Components_Reconstructed_{}.png'.format(path, i), format='png')

    def save_result(self, path):
        """
        Parameters:
        -----------
        path: string
            Path to save results.

        Returns:
        --------
        None
        """
        if not os.path.exists(path):
            os.makedirs(path)

        # self.save('{}/model_parameters.pickle'.format(path))
        self._write_results(path)
        self._save_input(path)
        self._save_model_reconstruction(path)
        self._save_representation(path)

    def create_annotated_plot(self, ax, X, vmin=0, vmax=1):
        ax.imshow(X, cmap='gray', vmin=vmin, vmax=vmax)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                ax.text(j, i, '{:.2g}'.format(X[i, j]), ha="center", va="center", color="w" if X[i, j] < vmin+vmax/2 else "k")
        return ax

    def save(self, file_name):
        """
        Save checkpoint.

        Parameters:
        -----------
        file_name: string
            Path to save results.

        Returns:
        --------
        None
        """
        fid = open(file_name, 'wb')
        pickle.dump(self.__dict__, fid)
        fid.close()


    def load(self, file_name):
        """
        Load checkpoint.

        Parameters:
        -----------
        file_name: string
            Path to save results.

        Returns:
        --------
        None
        """
        fid = open(file_name, 'rb')
        tmp_dict = pickle.load(fid)
        fid.close()

        self.__dict__.update(tmp_dict)

if __name__ == '__main__':
    # Create toy dataset
    pi = 4
    pj = 4

    X0 = np.random.rand(pi,pj)
    X1 = np.random.rand(pi,pj)
    X2 = np.random.rand(pi,pj)

    # Mix signals
    X = np.array([X0.flatten(), X1.flatten(), X2.flatten()])
    A = np.array([[1.5, 2, 3], [3, 0.5, 2], [2, 3.5, 0.5]])
    M = np.dot(A, X)
    M = M.reshape(M.shape[0], pi, pj)
    M /= np.max(M)

    # Run model
    N = 3
    ica = latent_linear_model(M, N, model='ica', order=3, multi_channel=False, device='cpu')
    ica.fit(lr=1, iterations=50, natural_gradient=True, tol=1e-7, verbose=True, verbose_step=1, store_values=False)

    # Reconstruct and create visualisation on the data layer
    X_reconstructed = ica.reconstruct_data()
    for i in range(M.shape[0]):
        fig, ax = plt.subplots(figsize=(15,15))
        ica.create_annotated_plot(ax, M[i])
        ax.set_title('Input')
        fig, ax = plt.subplots(figsize=(15,15))
        ica.create_annotated_plot(ax, X_reconstructed[i])
        ax.set_title('Reconstruction')
        fig, ax = plt.subplots(figsize=(15,15))
        ica.create_annotated_plot(ax, M[i] - X_reconstructed[i])
        ax.set_title('Absolute Reconstruction Error')

    # Reconstruct and create visualisation on the representation layer
    C = ica.reconstruct_components()
    X_array = [X0, X1, X2]

    for i in range(M.shape[0]):
        fig, ax = plt.subplots(figsize=(15,15))
        ica.create_annotated_plot(ax, X_array[i])
        ax.set_title('Input')

    # NOTE: The order of the input images cannot be recovered.
    for i in range(M.shape[0]):
        fig, ax = plt.subplots(figsize=(15,15))
        ica.create_annotated_plot(ax, C[i])
        ax.set_title('Reconstruction')

