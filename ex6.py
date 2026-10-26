import numpy as np
from matplotlib import pyplot as plt
from ex6_utils import (plot_ims, load_MNIST, outlier_data, gmm_data, plot_2D_gmm, load_dogs_vs_frogs,
                       BayesianLinearRegression, poly_kernel, cluster_purity)
from scipy.special import logsumexp
from typing import Tuple
import random


def outlier_regression(model: BayesianLinearRegression, X: np.ndarray, y: np.ndarray, p_out: float, T: int,
                       mu_o: float=0, sig_o: float=10) -> Tuple[BayesianLinearRegression, np.ndarray]:
    """
    Gibbs sampling algorithm for robust regression (i.e. regression assuming there are outliers in the data)
    :param model: the Bayesian linear regression that will be used to fit the data
    :param X: the training data, as a numpy array of shape [N, d] where N is the number of points and d is the dimension
    :param y: the regression targets, as a numpy array of shape [N,]
    :param p_out: the assumed probability for outliers in the data
    :param T: number of Gibbs sampling iterations to use in order to fit the model
    :param mu_o: the assumed mean of the outlier points
    :param sig_o: the assumed variance of the outlier points
    :return: the fitted model assuming outliers, as a BayesianLinearRegression model, as well as a numpy array of the
             indices of points which were considered as outliers
    """
    # Iinitnitialize outliers - assuming no outliers at start (all False)
    N = X.shape[0]
    outliers = np.zeros(N, dtype=bool) 
    
    
    # p(y | outlier) = N(y | mu_o, sig_o)
    log_prob_y_given_outlier = -0.5 * np.log(2 * np.pi * sig_o) - ((y - mu_o)**2) / (2 * sig_o)
    
    # Gibbs Sampling for T iterations
    for _ in range(T):
        inliers_indices = ~outliers
        
        # if there are inliers, fit the model.
        if np.sum(inliers_indices) > 0:
            model.fit(X[inliers_indices], y[inliers_indices], sample=True)
            
        
        # log likelihood of data being an inlier: log N(y | model(x), sigma^2)
        log_prob_y_given_inlier = model.log_likelihood(X, y)
        
      
        log_joint_out = np.log(p_out) + log_prob_y_given_outlier
        log_joint_in = np.log(1 - p_out) + log_prob_y_given_inlier
        

        prob_outlier = 1.0 / (1.0 + np.exp(log_joint_in - log_joint_out))
        
        
        outliers = np.random.rand(N) < prob_outlier

    
    return model, np.where(outliers)[0]


class BayesianGMM:
    def __init__(self, k: int, alpha: float, mu_0: np.ndarray, sig_0: float, nu: float, beta: float,
                 learn_cov: bool=True):
        """
        Initialize a Bayesian GMM model
        :param k: the number of clusters to use
        :param alpha: the value of alpha to use for the Dirichlet prior over the mixture probabilities
        :param mu_0: the mean of the prior over the means of each Gaussian in the GMM
        :param sig_0: the variance of the prior over the means of each Gaussian in the GMM
        :param nu: the nu parameter of the inverse-Wishart distribution used as a prior for the Gaussian covariances
        :param beta: the variance of the inverse-Wishart distribution used as a prior for the covariances
        :param learn_cov: a boolean indicating whether the cluster covariances should be learned or not
        """
        self.k = k
        self.alpha = alpha
        self.mu_0 = mu_0
        self.sig_0 = sig_0
        self.nu = nu
        self.beta = beta
        self.learn_cov = learn_cov
        self.d = mu_0.shape[0]


        self.pi = np.ones(k) / k
        

        self.means = np.random.randn(k, self.d) * np.sqrt(sig_0) + mu_0
        

        if learn_cov:

            self.covs = np.array([np.eye(self.d) for _ in range(k)])
        else:

            self.covs = None 

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the log-likelihood of each data point under each Gaussian in the GMM
        :param X: the data points whose log-likelihood should be calculated, as a numpy array of shape [N, d]
        :return: the log-likelihood of each point under each Gaussian in the GMM
        """
        N = X.shape[0]
        log_likes = np.zeros((N, self.k))
        
        if self.learn_cov:
            # Learned Covariance
            # log N(x | mu, Sigma) = -0.5*(d*ln(2pi) + ln|Sigma| + (x-mu)^T Sigma^-1 (x-mu))
            const_term = -0.5 * self.d * np.log(2 * np.pi)
            
            for j in range(self.k):
                try:
                    
                    L = np.linalg.cholesky(self.covs[j])
                    
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    
                    diff = X - self.means[j] # [N, d]
                    
                    y_solve = np.linalg.solve(L, diff.T)
                    mahalanobis = np.sum(y_solve**2, axis=0) 
                    
                    log_likes[:, j] = const_term - 0.5 * log_det - 0.5 * mahalanobis
                except np.linalg.LinAlgError:
                    log_likes[:, j] = -np.inf
            return log_likes

        #  Fixed Covariance
        #  log N(...) = -0.5 * [d*log(2pi*beta) + (1/beta)*||x-mu||^2]
        const_term = -0.5 * self.d * np.log(2 * np.pi * self.beta)
        for j in range(self.k):
            dist_sq = np.sum((X - self.means[j])**2, axis=1)
            log_likes[:, j] = const_term - (0.5 / self.beta) * dist_sq
            
        return log_likes

    def cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Clusters the data according to the learned GMM
        :param X: the data points to be clustered, as a numpy array of shape [N, d]
        :return: a numpy array containing the indices of the Gaussians the data points are most likely to belong to,
                 as a numpy array with shape [N,]
        """
        # log p(x|z=k)
        log_l = self.log_likelihood(X)
        
        
        log_posterior = log_l + np.log(self.pi + 1e-15)[None, :]
        
       
        return np.argmax(log_posterior, axis=1)

    def gibbs_fit(self, X:np.ndarray, T: int) -> 'BayesianGMM':
        """
        Fits the Bayesian GMM model using a Gibbs sampling algorithm
        :param X: the training data, as a numpy array of shape [N, d] where N is the number of points
        :param T: the number of sampling iterations to run the algorithm
        :return: the fitted model
        """
        N = X.shape[0]
        
        for t in range(T):
            
            log_l = self.log_likelihood(X)
            
            log_q = log_l + np.log(self.pi + 1e-15)[None, :]
            
            
            log_norm = logsumexp(log_q, axis=1, keepdims=True)
            probs = np.exp(log_q - log_norm)
            
            
            cum_probs = np.cumsum(probs, axis=1)
            rand_vals = np.random.rand(N, 1)
            z = np.argmax(cum_probs >= rand_vals, axis=1)
            
            
            counts = np.bincount(z, minlength=self.k)
            
           
            alpha_vec = self.alpha * np.ones(self.k) if np.ndim(self.alpha)==0 else self.alpha
            self.pi = np.random.dirichlet(alpha_vec + counts)
            
            
            prior_prec_val = 1.0 / self.sig_0
            prior_mean_vec = self.mu_0
            
            for j in range(self.k):
                indices = np.where(z == j)[0]
                N_k = counts[j]
                
                
                if N_k == 0:
                    self.means[j] = np.random.randn(self.d) * np.sqrt(self.sig_0) + self.mu_0
                    continue

                sum_x = np.sum(X[indices], axis=0)

                if self.learn_cov:
                    
                    Sigma_inv = np.linalg.inv(self.covs[j])
                    
                    
                    prior_prec_mat = prior_prec_val * np.eye(self.d)
                    post_prec = N_k * Sigma_inv + prior_prec_mat
                    post_cov = np.linalg.inv(post_prec)
                    
                    
                    mean_term = Sigma_inv @ sum_x + prior_prec_mat @ prior_mean_vec
                    post_mean = post_cov @ mean_term
                    
                    
                    self.means[j] = np.random.multivariate_normal(post_mean, post_cov)
                    
                else:

                    denom = (N_k / self.beta) + prior_prec_val
                    
                    
                    num = (1.0 / self.beta) * sum_x + prior_prec_val * prior_mean_vec
                    
                    post_mean = num / denom
                    post_var = 1.0 / denom
                    
                    
                    self.means[j] = np.random.randn(self.d) * np.sqrt(post_var) + post_mean

            
            if self.learn_cov:
                
                for j in range(self.k):
                    indices = np.where(z == j)[0]
                    N_k = counts[j]
                    
                    
                    prior_term = self.nu * np.eye(self.d) * self.beta
                    
                    
                    if N_k > 0:
                        diff = X[indices] - self.means[j] 
                        data_term = diff.T @ diff
                    else:
                        data_term = np.zeros((self.d, self.d))
                        
                    
                    self.covs[j] = (prior_term + data_term) / (self.nu + N_k)

        return self


if __name__ == '__main__':

    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    # ------------------------------------------------------ section 2 - Robust Regression
    # ---------------------- question 2
    # load the outlier data
    x, y = outlier_data(50)
    # init BLR model that will be used to fit the data
    mdl = BayesianLinearRegression(theta_mean=np.zeros(2), theta_cov=np.eye(2), sample_noise=0.15)

    # sample using the Gibbs sampling algorithm and plot the results
    plt.figure()
    plt.scatter(x, y, 15, 'k', alpha=.75)
    xx = np.linspace(-0.2, 5.2, 100)
    
    # Iterating over specified T values
    for t in [0, 1, 5, 10, 25]:
        samp, outliers = outlier_regression(mdl, x, y, T=t, p_out=0.1, mu_o=4, sig_o=2)
        plt.plot(xx, samp.predict(xx), lw=2, label=f'T={t}')
        
    plt.xlim([np.min(xx), np.max(xx)])
    plt.legend()
    plt.show()

    # ---------------------- question 3 
    # load the images to use for classification
    N = 1000
    ims, labs = load_dogs_vs_frogs(N)
    # define BLR model that should be used to fit the data
    mdl = BayesianLinearRegression(sample_noise=0.001, kernel_function=poly_kernel(2))
    # use Gibbs sampling to sample model and outliers
    samp, outliers = outlier_regression(mdl, ims, labs, p_out=0.01, T=50, mu_o=0, sig_o=.5)
    # plot the outliers
    plot_ims(ims[outliers], title='outliers')

    # ------------------------------------------------------ section 3 - Bayesian GMM
    # ---------------------- question 5
    # load 2D GMM data
    k, N = 5, 1000
    X = gmm_data(N, k)

    active_cluster_counts = []  # Initialize a list to store counts 

    for i in range(5):
        gmm = BayesianGMM(k=50, alpha=.01, mu_0=np.zeros(2), sig_0=.5, nu=5, beta=.5)
        gmm.gibbs_fit(X, T=100)

        
        # count how many mixing probabilities (pi) are greater than 10^-4
        count = np.sum(gmm.pi > 1e-4)
        active_cluster_counts.append(count)
        print(f"Run {i+1}: Found {count} active clusters")
        

        # plot a histogram of the mixture probabilities (in descending order)
        pi = gmm.pi  
        plt.figure()
        plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
        plt.ylabel(r'$\pi_k$')
        plt.xlabel('cluster number')
        plt.show()

        # plot the fitted 2D GMM
        plot_2D_gmm(X, gmm.means, gmm.covs, gmm.cluster(X))

    
    print(f"Average number of clusters with pi > 10^-4: {np.mean(active_cluster_counts)}")




    # # ---------------------- questions 6-7
    # # load image data
    # MNIST, labs = load_MNIST()
    # # flatten the images
    # ims = MNIST.copy().reshape(MNIST.shape[0], -1)
    
    # # learn_cov=False for images
    # gmm = BayesianGMM(k=500, alpha=1, mu_0=0.5*np.ones(ims.shape[1]), sig_0=.1, nu=1, beta=.25, learn_cov=False)
    # gmm.gibbs_fit(ims, 100)

    # # plot a histogram of the mixture probabilities (in descending order)
    # pi = gmm.pi
    # plt.figure()
    # plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
    # plt.ylabel(r'$\pi_k$')
    # plt.xlabel('cluster number')
    # plt.show()

    # # find the clustering of the images to different Gaussians
    # cl = gmm.cluster(ims)
    # clusters = np.unique(cl)
    # print(f'{len(clusters)} clusters used')
    
    # # calculate the purity of each of the clusters
    # purities = np.array([cluster_purity(labs[cl == k]) for k in clusters])
    # purity_inds = np.argsort(purities)

    # # plot 25 images from each of the clusters with the top 5 purities
    # for ind in purity_inds[-5:]:
    #     clust = clusters[ind]
    #     # ensure float for plotting
    #     plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')

    # # plot 25 images from each of the clusters with the bottom 5 purities
    # for ind in purity_inds[:5]:
    #     clust = clusters[ind]
    #     plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')


