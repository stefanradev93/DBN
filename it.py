import numpy as np


def Likelihood(s,z,vartheta_par):
    if z == 0:
        s = 1-s
    f = s/np.sum(s)
    h = vartheta_par*f + (1-vartheta_par)*np.array([1/3,1/3,1/3])
    return h

def Posterior(lik,pre):
    out = (lik*pre)/np.sum(lik*pre)
    return out

def smoothVec(z,s,epsilon_par,lambda_par,xi_par,omega0):
    omega = z*lambda_par*(omega0**epsilon_par)*s + (1-z)*xi_par*(1-(omega0**epsilon_par)*s) + (1 - (z*lambda_par + (1-z)*xi_par))*omega0
    return omega

def TransMat(o):
    G = np.diag(o)
    G[0][G[0]==0] = (1-o[0])/2
    G[1][G[1]==0] = (1-o[1])/2
    G[2][G[2]==0] = (1-o[2])/2
    return G

def ChapmanKolmogorov(G,posterior):
    p = np.inner(posterior,np.transpose(G))
    return p

def MatchingMatrix(n_trials, ambiguity):
    M = np.array([np.random.choice([1,2,3,4],3, replace=ambiguity) for _ in range(n_trials)])
    return M


def KL(prob_from,prob_to):
    """
    Compute KL-divergence between predictive distributions at trial t and t+1.
    """

    klv = np.array([np.sum(prob_to[i] * np.log(prob_to[i]/prob_from[i]) + 1e-15) for i in range(len(prob_from))])
    return klv

def surprise(lik,pre):
    """
    Compute Surprise between predictive distribution and likelihood at trial t.
    """

    s = np.array([-np.log(np.sum(lik[i]*pre[i])) for i in range(len(pre))])
    return s


def entropy(pre):
    """
    Compute the entropy of the predictive distribution at trial t.
    """

    h = np.array([-np.sum(pre[i] * np.log(pre[i])) for i in range(len(pre))])
    return h

def it_measures(D, M, params):
    """
    Computes KL, surprise, and entropy given data, matching signal and parameters.
    ----------

    Arguments:
    # D: observed data array [Z, Y]
    # M: Empirical Matching Matrix
    # params: parameter vector [epsilon, xi]

    """
    
    epsilon_par, xi_par = params
    vartheta_par=.99
    lambda_par=.99
    z = D[:,0]
    y = D[:,1]
    max_trials = len(M)
    s = np.zeros((max_trials,3), dtype=int) # Initialize the signal vector
    omega = np.zeros((max_trials,3))        # Initialize the vector of the attention to reward process
    pre = np.zeros((max_trials,3))          # Initialize the vector of the predictive (prior) probabilities
    lik = np.zeros((max_trials,3))          # Initialize the vector of likelihoods
    post = np.zeros((max_trials,3))         # Initialize the vector of posteriors
    pre[0] = np.array([1/3,1/3,1/3])
    s[0] = y[0] == M[0] 
    omega[0] = smoothVec(z[0],s[0],epsilon_par,lambda_par,xi_par,np.array([0.5,0.5,0.5]))
    lik[0] = Likelihood(s[0],z[0],vartheta_par)
    post[0] = Posterior(lik[0],pre[0])
    G = TransMat(omega[0])
    for t in range(1, max_trials):
        pre[t] = ChapmanKolmogorov(G, post[t-1]) # Compute predictive distribution
        s[t] = y[t] == M[t] # signal
        omega[t] = smoothVec(z[t],s[t],epsilon_par,lambda_par,xi_par,omega[t-1]) # Update omega
        lik[t] = Likelihood(s[t],z[t],vartheta_par) # Compute Likelihood of the observation given the states
        post[t] = Posterior(lik[t],pre[t]) # Compute posterior distribution over the hidden states
        G = TransMat(omega[t]) # Update Stability Matrix
    
    B = KL(pre[0:127],pre[1:128])
    I = surprise(lik[0:127],pre[0:127])
    H = entropy(pre[0:127])
    # KL-divergence, Surprise, Entropy 
    return np.c_[B,I,H]