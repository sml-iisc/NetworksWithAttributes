import numpy as np
import torch
import torch.nn as nn

"""
Implements the model that:
    1. Uses mixed-membership SBM
    2. Using sampling from a simplex in RBM
    3. Uses two step ELBO maximization: (i) Using autograd to update MMSBM parameters, (ii) Using contrastive divergence
       to update RBM parameters.
"""


def xavier_init(num_inputs, num_outputs):
    """
    :param num_inputs: Number of input units in the layer
    :param num_outputs: Number of output units in the layer
    :return mat: A matrix of size (num_outputs, num_inputs) initialized using Xavier's initialization scheme.
    """
    return np.random.normal(loc=0, scale=2.0 / (num_outputs + num_inputs), size=(num_outputs, num_inputs))


def lbeta(alpha, beta):
    """
    Computes the log Beta(alpha, beta) function
    """
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RBMMSBM(nn.Module):

    def __init__(self, num_nodes, num_comm, num_attr, alphas, betas):
        """
        :param num_nodes: Number of nodes
        :param num_comm: Number of communities
        :param num_attr: Number of attributes
        :param alphas: (K, K) matrix of prior alphas for B
        :param betas: (K, K) matrix of prior betas for B
        """
        super(RBMMSBM, self).__init__()

        self.num_nodes = num_nodes
        self.num_comm = num_comm
        self.num_attr = num_attr

        # Note that the alphas and betas must be positive and hence the version that
        # will be used will exponentiate the values of these variables first
        self.alphas = torch.tensor(alphas, dtype=torch.float32, requires_grad=False)
        self.betas = torch.tensor(betas, dtype=torch.float32, requires_grad=False)

        # Initialize the weights and biases for RBM
        self.w = xavier_init(self.num_attr, self.num_comm)
        self.b = np.zeros((self.num_attr, 1))
        self.c = np.zeros((self.num_comm, 1))

        self.grad_b = np.zeros(self.b.shape)
        self.grad_c = np.zeros(self.c.shape)
        self.grad_w = np.zeros(self.w.shape)

        # Define the parameters for posterior on block matrix. Due to positivity constraints, the actual parameter is
        # given by exp(self.alphas_post) and exp(self.betas_post)
        self.alphas_post = nn.Parameter(torch.tensor(alphas, dtype=torch.float32, requires_grad=True))
        self.betas_post = nn.Parameter(torch.tensor(betas, dtype=torch.float32, requires_grad=True))

        # Define the parameters for posterior Dirichlet distribution modeling the mixed-membership vectors. Note that
        # the parameter is given by exp(self.mu) due to positivity constraint
        self.mu = nn.Parameter(torch.tensor(np.zeros((self.num_nodes, self.num_comm)), dtype=torch.float32,
                                            requires_grad=True))

        # For a given edge (i, j), self.q[i, j, :] is a vector of size (self.num_comm,) which models the posterior
        # probability of node i being in each of the self.num_comm classes. Again, the actual parameter is obtained by
        # taking the soft-max of the stored parameters due to the simplex constraint.
        self.q = nn.Parameter(torch.tensor(np.ones((self.num_nodes, self.num_nodes, self.num_comm)) / self.num_comm,
                                           dtype=torch.float32, requires_grad=True))

        # Initialize samples from RBM for persistent CD
        self.y_samples = None

    def forward(self, adj_mat, attr_mat):
        """
        :param adj_mat: (self.num_nodes, self.num_nodes) binary adjacency matrix of network
        :param attr_mat: (self.num_nodes, self.num_attr) binary attribute matrix of the nodes
        :return neg_elbo: Negative of ELBO
        """
        return self.e_step(adj_mat, attr_mat)

    def e_step(self, adj_mat, attr_mat):
        """
        :param adj_mat: (self.num_nodes, self.num_nodes) binary adjacency matrix of network
        :param attr_mat: (self.num_nodes, self.num_attr) binary attribute matrix of the nodes
        :return neg_elbo: Negative of ELBO
        """
        # Compute E[ln p(B)]
        alphas = torch.exp(self.alphas)
        betas = torch.exp(self.betas)
        alphas_post = torch.exp(self.alphas_post)
        betas_post = torch.exp(self.betas_post)
        di_alpha = torch.digamma(alphas_post)
        di_beta = torch.digamma(betas_post)
        di_sum = torch.digamma(alphas_post + betas_post)
        ln_pb = (alphas - 1) * (di_alpha - di_sum) + (betas - 1) * (di_beta - di_sum)
        ln_pb = ln_pb[torch.ones(ln_pb.size()).triu() == 1].sum()

        # Compute E[ln p(\bar{Z} | Z)]
        q = torch.softmax(self.q, dim=-1)
        mu = torch.exp(self.mu)
        di_mu = torch.digamma(mu)
        di_sum = torch.digamma(mu.sum(dim=1))
        di_diff = di_mu - di_sum.unsqueeze(1).expand_as(di_mu)
        di_diff = di_diff.unsqueeze(1).repeat(1, self.num_nodes, 1)
        ln_pbarz = (q * di_diff).sum(dim=2)
        ln_pbarz = ln_pbarz[torch.ones(ln_pbarz.size()) - torch.eye(self.num_nodes) == 1].sum()

        # Compute E[ln p(Z | Y)]
        mu_normalized = mu / (mu.sum(dim=1).unsqueeze(1).expand_as(mu))
        w = torch.from_numpy(self.w.T).float()
        b = torch.from_numpy(self.b).float()
        c = torch.from_numpy(self.c).float()
        ln_pz = (w.matmul(mu_normalized.t()).t() * attr_mat).sum()
        ln_pz += attr_mat.matmul(b).sum()
        ln_pz += (c * mu_normalized.sum(dim=0).view(-1, 1)).sum()

        # Compute ln P(A | \bar{Z}, B)
        ln_pa = 0.0
        di_sum = torch.digamma(alphas_post + betas_post)
        for i in range(self.num_nodes):
            for j in range(i):
                temp = q[i, j, :].view(-1, 1).matmul(q[j, i, :].view(1, -1))
                temp += q[j, i, :].view(-1, 1).matmul(q[i, j, :].view(1, -1))
                if adj_mat[i, j] == 1:
                    temp = temp * (di_alpha - di_sum)
                else:
                    temp = temp * (di_beta - di_sum)
                temp = temp.sum()
                ln_pa += temp

        # Compute entropy of q(B)
        ent_b = lbeta(alphas_post, betas_post) - (alphas_post - 1) * di_alpha - (betas_post - 1) * di_beta + \
            (alphas_post + betas_post - 2) * di_sum
        ent_b = ent_b[torch.ones(ent_b.size()).triu() == 1].sum()

        # Compute entropy of q(\bar{Z})
        ln_q = torch.log(1e-6 + q)
        ent_barz = (q * ln_q).sum(dim=2)[torch.ones(self.num_nodes, self.num_nodes) -
                                         torch.eye(self.num_nodes) == 1].sum()
        ent_barz = -ent_barz

        # Compute entropy of q(Z)
        mu0 = mu.sum(dim=1)
        ent_z = (torch.lgamma(mu).sum(dim=1) - torch.lgamma(mu0)).sum()
        ent_z += ((mu0 - self.num_comm) * torch.digamma(mu0)).sum()
        ent_z -= ((mu - 1) * torch.digamma(mu)).sum()

        # Compute ELBO
        elbo = ln_pb + ln_pbarz + ln_pz + ln_pa + ent_b + ent_barz + ent_z
        elbo = -elbo

        return elbo

    def m_step(self, attr_mat, num_samples=1, chain_length=10, lr=1e-2, momentum=0.0, use_persistence=False):
        """
        :param attr_mat: (self.num_nodes, self.num_comm) observed binary feature matrix
        :param num_samples: Number of samples to use for approximating expectations
        :param chain_length: Length of Gibbs chain for RBM sampling
        :param lr: Learning rate for parameter updates
        :param momentum: Momentum term for SGD update
        :param use_persistence: Whether to use persistent-CD or not
        """
        # Get samples from RBM
        if not use_persistence or self.y_samples is None:
            self.y_samples = np.asarray(attr_mat[np.random.choice(self.num_nodes, size=num_samples), :])
        y, z = self.rbm_sample(chain_length=chain_length, num_samples=num_samples, y=self.y_samples)
        self.y_samples = y

        # Compute gradients for RBM parameters
        mu = torch.exp(self.mu)
        mu_normalized = mu / (mu.sum(dim=1).unsqueeze(1).expand_as(mu))
        mu_normalized = mu_normalized.detach().numpy()
        grad_b = (-self.num_nodes * y.mean(axis=0) + attr_mat.sum(axis=0)).reshape((-1, 1))
        grad_c = (-self.num_nodes * z.mean(axis=0) + mu_normalized.sum(axis=0)).reshape((-1, 1))
        grad_w = -self.num_nodes * (np.matmul(y.T, z) / num_samples) + np.matmul(attr_mat.T, mu_normalized)

        # Update the RBM parameters
        self.grad_b = momentum * self.grad_b + (1 - momentum) * grad_b / self.num_nodes
        self.b += lr * self.grad_b
        self.grad_c = momentum * self.grad_c + (1 - momentum) * grad_c / self.num_nodes
        self.c += lr * self.grad_c
        self.grad_w = momentum * self.grad_w + (1 - momentum) * grad_w.T / self.num_nodes
        self.w += lr * self.grad_w

        self.w = self.w.clip(-5, 5)
        self.b = self.b.clip(-5, 5)
        self.c = self.c.clip(-5, 5)

    def rbm_sample(self, y=None, z=None, chain_length=10, num_samples=1, start_with_z=True):
        """
        y: (num_samples, M) binary starting point for observable features
        z: (num_samples, K) simplex starting point for mixed membership
        chain_length: Number of steps to take for Gibbs sampling
        num_samples: Number of samples to generate
        start_with_z: Whether to start the Gibbs chain by sampling z
        returns: y, z
            y: (num_samples, M) Sampled y values
            z: (num_samples, K) Sampled z values
        """
        if y is None:
            y = (np.random.random(size=(num_samples, self.num_attr)) <= 0.5).astype(float)
        if z is None:
            z = np.random.dirichlet(np.random.rand(self.num_comm), size=num_samples)

        c = self.c
        b = self.b
        w = self.w.T
        for _ in range(chain_length):
            if start_with_z:
                for k in range(self.num_comm - 1):
                    yw = np.matmul(y, w)
                    betak = c[k, 0] - c[-1, 0] + yw[:, k] - yw[:, -1]
                    sumk = z[:, k] + z[:, -1]
                    u = np.random.random((num_samples,))
                    zk = np.log(u * (np.exp(sumk * betak) - 1) + 1) / (1e-6 + betak)
                    z[:, k] = zk
                    z[:, -1] = sumk - zk
                y = sigmoid(np.matmul(z, w.T) + b.repeat(num_samples, axis=1).T)
                y = (np.random.random(y.shape) <= y).astype(float)
            else:
                y = sigmoid(np.matmul(z, w.T) + b.repeat(num_samples, axis=1).T)
                y = (np.random.random(y.shape) <= y).astype(float)
                for k in range(self.num_comm - 1):
                    yw = np.matmul(y, w)
                    betak = c[k, 0] - c[-1, 0] + yw[:, k] - yw[:, -1]
                    sumk = z[:, k] + z[:, -1]
                    u = np.random.random((num_samples,))
                    zk = np.log(u * (np.exp(sumk * betak) - 1) + 1) / (1e-6 + betak)
                    z[:, k] = zk
                    z[:, -1] = sumk - zk
        return y, z
