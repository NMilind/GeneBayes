from functools import partial

import numpy as np

import torch
import torch.distributions as dists

from torchquad import set_up_backend, Boole

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


torch.set_default_dtype(torch.float64)
set_up_backend('torch', data_type='float64')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print = partial(print, flush=True)

# What follows is analytical example with the following distributions
# U | X ~ N(m, s^2)
# Y | U ~ N(U, sigma^2)
# Y | X ~ N(m, sigma^2 + s^2) = integral of Pr(Y | U) Pr(U | X) dU
# U | Y ~ N((sigma^2 / (s^2 + sigma^2)) m + (s^2 / (s^2 + sigma^2)) Y, (s^2 sigma^2) / (s^2 + sigma^2)))

#----------------------------------------------------------
# Likelihood
#----------------------------------------------------------

def log_likelihood(y, sigma_2, u):
    
    '''
    The log likelihood of the data, which is Pr(Y | U).
    '''

    log_lik = dists.Normal(loc=u, scale=torch.sqrt(sigma_2)).log_prob(y)

    return log_lik

#----------------------------------------------------------
# Prior
#----------------------------------------------------------

def log_prior(u, m, s_2):

    '''
    The prior, which is Pr(U | X) = Pr(U | m(X), s_2(X)).
    '''

    log_pr = dists.Normal(loc=m, scale=torch.sqrt(s_2)).log_prob(u)

    return log_pr

#----------------------------------------------------------
# Evidence
#----------------------------------------------------------

def evidence(y, sigma_2, m, s_2, u):

    '''
    The evidence, which is Pr(Y | X) = Pr(Y | U) * Pr(U | X).
    '''

    log_pr = log_prior(u, m, s_2)
    log_lik = log_likelihood(y, sigma_2, u)

    return torch.exp(log_pr + log_lik)

#----------------------------------------------------------
# Evidence Integral
#----------------------------------------------------------

# Example input tensors for JIT trace
eg_y = torch.tensor(0.0)
eg_sigma_2 = torch.tensor(1.0)
eg_m = torch.tensor(0.0)
eg_s_2 = torch.tensor(1.0)

# Transform so that U is in [0, 1] rather than [-infinity, infinity]
def evidence_transformed(y, sigma_2, m, s_2, u):

    eval_1 = evidence(y, sigma_2, m, s_2, (1 / u) - 1)
    eval_2 = evidence(y, sigma_2, m, s_2, (-1 / u) + 1)
    
    return (eval_1 + eval_2) / torch.pow(u, 2)

# The endpoints (0 and 1) are undefined, so we get arbitrarily close
INT_LB = 1E-20
INT_UB = 1
domain = torch.tensor([[INT_LB, INT_UB]])

# Define a grid with a set number of integration points
boole = Boole()
N_INTEGRATION_PTS = 1001
grid_points, hs, n_per_dim = boole.calculate_grid(N_INTEGRATION_PTS, domain)
grid_size = (INT_LB - INT_UB) / N_INTEGRATION_PTS

def jit_integrate_evidence(y, sigma_2, m, s_2):

    evals, _ = boole.evaluate_integrand(
        partial(evidence_transformed, y, sigma_2, m, s_2),
        grid_points
    )
    return boole.calculate_result(evals, 1, n_per_dim, hs)

integrate_evidence = torch.jit.trace(
    jit_integrate_evidence,
    (eg_y, eg_sigma_2, eg_m, eg_s_2)
)

#----------------------------------------------------------
# Score
#----------------------------------------------------------

# The score is used for Gradient Boosting. GeneBayes uses
# the negative log evidence as the score, which results in
# Maximum Likelihood Estimation (MLE).

class PriorLogScore(LogScore):
    
    def score(self, D):

        '''
        The negative log evidence of the data.

        Note that the evidence is -log Pr(Y | X).

        :param D: All the observed data.
        '''

        # Convert to Torch tensors for gradient calculation
        Y = torch.tensor(D, device=device)
        M = torch.tensor(self.m, requires_grad=True, device=device)
        S_2 = torch.tensor(self.s_2, requires_grad=True, device=device)
        SIGMA_2 = torch.tensor(self.sigma_2, device=device)

        # Aggregate the score for all data into one value
        score = torch.tensor(0.0, device=device)

        # Create a matrix to store the gradient w.r.t. each parameter
        self.gradient = torch.zeros(Y.shape[0], self.n_params, device=device)

        # Iterate over each data point and calculate score
        for y, sigma_2, m, s_2 in zip(Y.split(1), SIGMA_2.split(1), M.split(1), S_2.split(1)):

            # Calculate the evidence
            Pr_Y = integrate_evidence(y, sigma_2, m, s_2)
            assert Pr_Y.item() >= 0, f'Invalid likelihood: {Pr_Y.item()} with parameters {y.item()}, {sigma_2.item()}, {m.item()}, {s_2.item()}'

            # Add the negative log evidence (score)
            score += -torch.log(Pr_Y)

        # Calculate gradients of parameters based on score
        score.backward()

        # Store gradients
        # NOTE: For S_2, multiply by (dlog(s_2) / ds_2)^(-1)
        self.gradient[:, 0] = M.grad
        self.gradient[:, 1] = S_2.grad * S_2
        self.gradient = self.gradient.detach().cpu().numpy()

        # Set gradients to zero (clear buffers)
        M.grad = None
        S_2.grad = None

        return score.item()
    
    def d_score(self, D):

        '''
        The derivative of the score (negative log evidence).

        :param U: All the hidden variables U.
        '''
        
        # Log current parameter values
        for i in range(Prior.n_params):
            p0 = np.min(self._params[i])
            p1 = np.quantile(self._params[i], 0.01)
            p99 = np.quantile(self._params[i], 0.99)
            p100 = np.max(self._params[i])
            print(f'Parameter {i} - Min: {p0}, 1st percentile: {p1}, 99th percentile: {p99}, max: {p100}')

        # Calculate score to recalculate gradients
        self.score(D)

        return self.gradient

    def metric(self, n_mc_samples=100):

        '''
        Calculate the Riemannian Metric for the negative log evidence
        (a.k.a. Fisher information metric).
        '''

        all_grad = list()

        # Iterate for MC samples
        for _ in range(n_mc_samples):

            # Create tensors from parameters
            M = torch.tensor(self.m)
            S = torch.tensor(self.s)
            SIGMA = torch.tensor(self.sigma)
            
            # Sample Y | X by sampling U | X and then Y | U
            prior = dists.Normal(M, S)
            u = prior.sample()
            # NOTE: Check in with Tony/Jeff about this, might need to sample from U | X instead
            likelihood = dists.Normal(u, SIGMA)
            y = likelihood.sample().numpy()

            # Calculate gradients and add
            all_grad.append(self.d_score(y))

        # Calculate Fisher Information Metric matrix
        grad = np.stack(all_grad)
        grad = np.mean(np.einsum('sik,sij->sijk', grad, grad), axis=0)

        return grad


#----------------------------------------------------------
# Distribution
#----------------------------------------------------------

# The distributional form of the response. In our case,
# this is Pr(Y | X) = integral of Pr(Y | U) Pr(U | X) dU

class Prior(RegressionDistn):

    n_params = 2
    scores = [PriorLogScore]
    multi_output = False

    def __init__(self, params, fixed):

        # Internal reference required for NGBoost
        self._params = params
        self._fixed = fixed

        # Extract parameters
        self.m = params[0]
        self.log_s_2 = params[1]

        # Extract fixed parameters
        self.sigma_2 = fixed[0]

        # Convert to useful forms
        self.s_2 = np.exp(self.log_s_2)
        self.s = np.sqrt(self.s_2)
        self.sigma = np.sqrt(self.sigma_2)

    def fit(D, F):

        '''
        Fit the observations to the marginal distribution. In NGBoost, this is used to
        initialize the parameters before using boosting to better fit the data based on
        the gradients of the score.

        :param D: All the observed data.
        '''

        Y = torch.tensor(D, device=device)
        log_sigma_2 = torch.tensor(F[0], device=device)

        # Initialize parameters
        m_y = torch.tensor(0.0, requires_grad=True, device=device)
        log_s_2_y = torch.tensor(1.0, requires_grad=True, device=device)

        # Initialize optimizer
        LR_INIT = 0.001
        optimizer = torch.optim.AdamW([m_y, log_s_2_y], lr=LR_INIT)

        # Expand parameters to have one for each data point
        m_y = m_y.expand(Y.shape[0])
        log_s_2_y = log_s_2_y.expand(Y.shape[0])

        # Iterate over a set number of epochs
        MAX_EPOCHS_INIT = 150

        for i in range(MAX_EPOCHS_INIT):
            
            loss = 0

            # In each epoch, optimize over each data point individually
            # Essentially, stochastic gradient descent
            for idx in torch.randperm(Y.shape[0]):
                
                # Zero out buffers
                optimizer.zero_grad()

                # Extract the parameter values for this data point
                # NOTE: The parameter value is shared across all data points
                y = Y[idx]
                sigma_2 = torch.exp(log_sigma_2[idx])
                m = m_y[idx]
                s_2 = torch.exp(log_s_2_y[idx])

                # Calculate negative log evidence as the loss
                result = -torch.log(integrate_evidence(y, sigma_2, m, s_2))
        
                # Calculate gradients based on loss
                result.backward()

                # Update parameters based on gradients
                optimizer.step()

                # Update combined loss
                loss += result.item()

            print(f'{i + 1} | Loss: {loss} | Mean: {m_y[0].item()} | Var: {torch.exp(log_s_2_y[0]).item()}')

        return np.array([m_y[0].item(), log_s_2_y[0].item()])
    
    def mean(self):

        return self.m
    
    @property
    def params(self):

        '''
        Return the parameters of this distribution.
        '''

        return {'m': self.m, 's': self.s}


#----------------------------------------------------------
# Posterior Integral
#----------------------------------------------------------

def posterior(pr_y, y, sigma_2, m, s_2, u):

    '''
    Returns the posterior Pr(Y, U) / p(Y)
    '''

    pr_y_u = evidence(y, sigma_2, m, s_2, u)
    return torch.exp(torch.log(pr_y_u) - torch.log(pr_y))

# The endpoints (0 and 1) are undefined, so we get arbitrarily close
INT_POST_LB = -50.0
INT_POST_UB = 50.0
post_domain = torch.tensor([[INT_POST_LB, INT_POST_UB]])

N_INTEGRATION_PTS = 1001
boole = Boole()
grid_points, hs, n_per_dim = boole.calculate_grid(N_INTEGRATION_PTS, post_domain)
grid_size = (INT_POST_UB - INT_POST_LB) / N_INTEGRATION_PTS

eg_y = torch.tensor(0.0)
eg_sigma_2 = torch.tensor(1.0)
eg_m = torch.tensor(0.0)
eg_s_2 = torch.tensor(1.0)
eg_pr_y = torch.tensor(1.0)

boole = Boole()
integrate_posterior = torch.jit.trace(
    lambda pr_y, y, sigma_2, m, s_2: boole.evaluate_integrand(
        partial(posterior, pr_y, y, sigma_2, m, s_2),
        grid_points
    ),
    (eg_pr_y, eg_y, eg_sigma_2, eg_m, eg_s_2)
)

#----------------------------------------------------------
# Posterior Inference
#----------------------------------------------------------

def posterior_summary(X, Y, F, ngb):

    '''
    Calculate posterior summaries.

    :param X: The features X.
    :param Y: The response Y.
    :param ngb: The trained model.
    '''

    Y = torch.tensor(Y, device=device)
    F = torch.tensor(F, device=device)
    X_dist = ngb.pred_dist(X, F, max_iter=ngb.best_val_loss_itr)
    M = torch.tensor(X_dist.params['m'], device=device)
    S = torch.tensor(X_dist.params['s'], device=device)
    SIGMA_2 = F[:, 0].clone()

    prior_mean = X_dist.mean()

    post_mean = list()
    post_lower = list()
    post_upper = list()

    for y, sigma_2, m, s in zip(Y.split(1), SIGMA_2.split(1), M.split(1), S.split(1)):

        s_2 = torch.pow(s, 2)

        # Calculate Pr(Y)
        Pr_Y = integrate_evidence(y, sigma_2, m, s_2)

        # Calculate Pr(U | Y)
        post_pdf, _ = integrate_posterior(Pr_Y, y, sigma_2, m, s_2)
        post_pdf = post_pdf.squeeze()

        # Estimate E(U | Y)
        mean = boole.calculate_result(post_pdf * grid_points.squeeze(), 1, n_per_dim, hs).item()

        # Get 90% Credible Interval
        cdf = torch.cumulative_trapezoid(post_pdf, dx=grid_size)
        lower = (INT_POST_LB + (torch.sum(cdf < 0.05) + 1) * grid_size).item()
        upper = (INT_POST_LB + (torch.sum(cdf <= 0.95) + 1) * grid_size).item()

        post_mean.append(mean)
        post_lower.append(lower)
        post_upper.append(upper)

    post_mean = np.array(post_mean)
    post_lower = np.array(post_lower)
    post_upper = np.array(post_upper)

    return prior_mean, post_mean, post_lower, post_upper
