import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal, Normal, Cauchy, Chi2
import torch.autograd as autograd

import matplotlib.pyplot as plt
from network import *
from data_loader import *
import seaborn as sns

class fgan(object):
    """
    This class ensembles data generating process of Huber's contamination model and training process
    for estimating center parameter via F-GAN.

    Usage:
        >> f = fgan(p=100, eps=0.2, device=device, tol=1e-5)
        >> f.dist_init(true_type='Gaussian', cont_type='Gaussian', 
            cont_mean=5.0, cont_var=1.)
        >> f.data_init(train_size=50000, batch_size=500)
        >> f.net_init(d_hidden_units=[20], elliptical=False, activation_D1='LeakyReLU')
        >> f.optimizer_init(lr_d=0.2, lr_g=0.02, d_steps=5, g_steps=1)
        >> f.fit(floss='js', epochs=150, avg_epochs=25, verbose=50, show=True)

    Please refer to the Demo.ipynb for more examples.
    """
    def __init__(self, p, eps, device=None, tol=1e-5):

        """Set parameters for Huber's model epsilon
                X i.i.d ~ (1-eps) P(mu, Sigma) + eps Q, 
            where P is the real distribution, mu is the center parameter we want to 
            estimate, Q is the contamination distribution and eps is the contamination
            ratio.

        Args:
            p: dimension.
            eps: contamination ratio.
            tol: make sure the denominator is not zero.
            device: If no device is provided, it will automatically choose cpu or cuda.
        """

        self.p = p
        self.eps = eps
        self.tol = tol
        self.device = device if device is not None \
                      else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def dist_init(self, true_type='Gaussian', cont_type='Gaussian', 
                  cont_mean=None, cont_var=1, cont_covmat=None):

        """
        Set parameters for distribution under Huber contaminaton models. We assume
        the center parameter of the true distribution mu is 0 and the covariance
        is indentity martix. 

        Args:
            true_type : Type of real distribution P. 'Gaussian', 'Cauchy'.
            cont_type : Type of contamination distribution Q, 'Gaussian', 'Cauchy'.
            cont_mean: center parameter for Q
            cont_var: If scatter (covariance) matrix of Q is diagonal, cont_var gives 
                      the diagonal element.
            cont_covmat: Other scatter matrix can be provided (as torch.tensor format).
                         If cont_covmat is not None, cont_var will be ignored. 
        """

        self.true_type = true_type
        self.cont_type = cont_type
        
        ## settings for true distribution sampler
        self.true_mean = torch.zeros(self.p)  
        if true_type == 'Gaussian':
            self.t_d = MultivariateNormal(torch.zeros(self.p), 
                                          covariance_matrix=torch.eye(self.p))
        elif true_type == 'Cauchy':
            self.t_normal_d = MultivariateNormal(torch.zeros(self.p), 
                                                 covariance_matrix=torch.eye(self.p))
            self.t_chi2_d = Chi2(df=1)
        else:
            raise NameError('True type must be Gaussian or Cauchy!')
        
        ## settings for contamination distribution sampler
        if cont_covmat is not None: 
            self.cont_covmat = cont_covmat
        else:
            self.cont_covmat = torch.eye(self.p) * cont_var
        self.cont_mean = torch.ones(self.p) * cont_mean
        if cont_type == 'Gaussian':
            self.c_d = MultivariateNormal(torch.zeros(self.p), 
                                          covariance_matrix=self.cont_covmat)
        elif cont_type == 'Cauchy':
            self.c_normal_d = MultivariateNormal(torch.zeros(self.p), 
                                                 covariance_matrix=self.cont_covmat)
            self.c_chi2_d = Chi2(df=1)
        else:
            raise NameError('Cont type must be Gaussian or Cauchy!')
    
    def _sampler(self, n):
        """ Sampler and it will return a [n, p] torch tensor. """

        if self.true_type == 'Gaussian':
            t_x = self.t_d.sample((n, ))
        elif self.true_type == 'Cauchy':
            t_normal_x = self.t_normal_d.sample((n, ))
            t_chi2_x = self.t_chi2_d.sample((n,))
            t_x = t_normal_x / (torch.sqrt(t_chi2_x.view(-1, 1)) + self.tol)

        if self.cont_type == 'Gaussian':
            c_x = self.c_d.sample((n, )) + self.cont_mean.view(1, -1)
        elif self.cont_type == 'Cauchy':
            c_normal_x = self.c_normal_d.sample((n, ))
            c_chi2_x = self.c_chi2_d.sample((n,))
            c_x = c_normal_x / (torch.sqrt(c_chi2_x.view(-1, 1)) + self.tol) +\
                  self.cont_mean.view(1, -1)

        s = (torch.rand(n) < self.eps).float()
        x = (t_x.transpose(1,0) * (1-s) + c_x.transpose(1,0) * s).transpose(1,0)

        return x

    def data_init(self, train_size=50000, batch_size=100):
        self.Xtr = self._sampler(train_size)
        self.batch_size = batch_size
        self.poolset = PoolSet(self.Xtr)
        self.dataloader = DataLoader(self.poolset, batch_size=self.batch_size, shuffle=True)
        
    def net_init(self, d_hidden_units, elliptical=False, 
                 g_input_dim=10, g_hidden_units=[10, 10],
                 activation_D1='Sigmoid', verbose=True):

        """
        Settings for Discriminator and Generator.

        Args:
            d_hidden_units: a list of hidden units for Discriminator, 
                            e.g. d_hidden_units=[10, 5], then the discrimintor has
                            structure p (input) - 10 - 5 - 1 (output).
            elliptical: Boolean. If elliptical == False, 
                            G_1(x|b) = x + b,
                        where b will be learned and x ~ Gaussian/Cauchy(0, I_p) 
                        according to the true distribution.
                        If elliptical = True,
                            G_2(t, u|b) = g_2(t)u + b,
                        where G_2(t, x|b) generates the family of elliptical 
                        distribution, t ~ Normal(0, I) and u ~ Uniform(\\|u\\|_2 = 1)
            g_input_dim: (Even) number. When elliptical == True, the dimension of input for 
                         g_2(t) need to be provided. 
            g_hidden_units: A list of hidden units for g_2(t). When elliptical == True, 
                            structure of g_2(t) need to be provided. 
                            e.g. g_hidden_units = [24, 12, 8], then g_2(t) has structure
                            g_input_dim - 24 - 12 - 8 - p.
            activation_D1: 'Sigmoid', 'ReLU' or 'LeakyReLU'. The first activation 
                            function after the input layer. Especially when 
                            true_type == 'Cauchy', Sigmoid activation is preferred.
            verbose: Boolean. If verbose == True, initial error 
                        \\|\\hat{\\mu}_0 - \\mu\\|_2
                     will be printed.
        """
        self.elliptical = elliptical
        self.g_input_dim = g_input_dim

        if self.elliptical:
            assert (g_input_dim % 2 == 0), 'g_input_dim should be an even number'
            self.netGXi = GeneratorXi(
                            input_dim=g_input_dim, 
                            hidden_units=g_hidden_units).to(self.device)
        self.netD = Discriminator(
                        p=self.p, hidden_units=d_hidden_units, 
                        activation_1=activation_D1).to(self.device)
        self.netG = Generator(p=self.p, elliptical=self.elliptical).to(self.device)

        # Initialize center parameter with sample median.
        self.netG.bias.data = torch.median(self.Xtr, dim=0)[0].to(self.device)
        self.mean_err_init = np.linalg.norm(self.netG.bias.data.cpu().numpy() -\
                                            self.true_mean.numpy())
        if verbose:
            print('Initialize Mean Error: %.4f' % self.mean_err_init)

        ## Initialize discrminator and g_2(t) when ellpitical == True
        self.netD.apply(weights_init_xavier)
        if (self.elliptical):
            self.netGXi.apply(weights_init_xavier)

    def optimizer_init(self, lr_d, lr_g, d_steps, g_steps):
        """
        Settings for optimizer.

        Args:
            lr_d: learning rate for discrimintaor.
            lr_g: learning rate for generator.
            d_steps: number of steps of discriminator per discriminator iteration.
            g_steps: number of steps of generator per generator iteration.

        """
        self.optG = optim.SGD(self.netG.parameters(), lr=lr_g)
        if self.elliptical:
            self.optGXi = optim.SGD(self.netGXi.parameters(), lr=lr_g)
        self.optD = optim.SGD(self.netD.parameters(), lr=lr_d)
        self.g_steps = g_steps    
        self.d_steps = d_steps

    def fit(self, floss='js', epochs=20, avg_epochs=10, use_inverse_gaussian=True,
            verbose=25, show=True):
        """
        Training process.
        
        Args:
            floss: 'js' or 'tv'. For JS-GAN, we consider the original GAN with 
                   Jensen-Shannon divergence and for TV-GAN, total variation will be
                   used.
            epochs: Number. Number of epochs for training.
            avg_epochs: Number. An average estimation using the last certain epochs.
            use_use_inverse_gaussian: Boolean. If elliptical == True, \\xi generator,
                                  g_2(t) takes random vector t as input and outputs
                                  \\xi samples. If use_use_inverse_gaussian == True, we take
                                  t = (t1, t2), where t1 ~ Normal(0, I_(d/2)) and
                                  t2 ~ 1/Normal(0, I_(d/2)), 
                                  otherwise, t ~ Normal(0, I_d).
            verbose: Number. Print intermediate result every certain epochs.
            show: Boolean. If show == True, final result will be printed after training.
        """
        assert floss in ['js', 'tv'], 'floss must be \'js\' or \'tv\''
        if floss == 'js':
            criterion = nn.BCEWithLogitsLoss()
        self.loss_D = [] 
        self.loss_G = []
        self.mean_err_record = []
        self.mean_est_record = []
        current_d_step = 1

        z_b = torch.zeros(self.batch_size, self.p).to(self.device)
        one_b = torch.ones(self.batch_size).to(self.device)
        if self.elliptical:
            if use_inverse_gaussian:
                xi_b1 = torch.zeros(self.batch_size, 
                                    self.g_input_dim//2).to(self.device)
                xi_b2 = torch.zeros(self.batch_size, 
                                    self.g_input_dim//2).to(self.device)
            else:
                xi_b = torch.zeros(self.batch_size, 
                                   self.g_input_dim).to(self.device)
        for ep in range(epochs):
            loss_D_ep = []
            loss_G_ep = []
            for _, data in enumerate(self.dataloader):
                ## update D
                self.netD.train()
                self.netD.zero_grad()
                ## discriminator loss
                x_real = data.to(self.device)
                feat_real, d_real_score = self.netD(x_real)
                if (floss == 'js'):
                    d_real_loss = criterion(d_real_score, one_b)
                elif floss == 'tv':
                    d_real_loss = - torch.sigmoid(d_real_score).mean()
                d_real_loss = criterion(d_real_score, one_b)
                ## generator loss
                if self.elliptical:
                    z_b.normal_()
                    z_b.div_(z_b.norm(2, dim=1).view(-1, 1) + self.tol)
                    if use_inverse_gaussian:
                        xi_b1.normal_()
                        xi_b2.normal_()
                        xi_b2.data = 1/(torch.abs(xi_b2.data) + self.tol)
                        xi = self.netGXi(
                            torch.cat([xi_b1, xi_b2], dim=1)).view(self.batch_size, -1)
                    else:
                        xi_b.normal_()
                        xi = self.netGXi(xi_b).view(self.batch_size, -1)
                    x_fake = self.netG(z_b, xi).detach()
                elif (self.true_type == 'Cauchy'):
                    z_b.normal_()
                    z_b.data.div_(torch.sqrt(
                                    self.t_chi2_d.sample((self.batch_size, 1))
                                            ).to(self.device) + self.tol
                                 )
                    x_fake = self.netG(z_b).detach()
                elif self.true_type == 'Gaussian':
                    x_fake = self.netG(z_b.normal_()).detach()
                feat_fake, d_fake_score = self.netD(x_fake)
                if floss == 'js':
                    d_fake_loss = criterion(d_fake_score, 1-one_b)
                elif floss == 'tv':
                    d_fake_loss = torch.sigmoid(d_fake_score).mean()
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                loss_D_ep.append((d_real_loss+d_fake_loss).cpu().item())
                self.optD.step()
                if current_d_step < self.d_steps:
                    current_d_step += 1
                    continue
                else:
                    current_d_step = 1

                ## update G
                self.netD.eval()
                for _ in range(self.g_steps):
                    self.netG.zero_grad()
                    if self.elliptical:
                        self.netGXi.zero_grad()
                        z_b.normal_()
                        z_b.div_(z_b.norm(2, dim=1).view(-1, 1) + self.tol)
                        if use_inverse_gaussian:
                            xi_b1.normal_()
                            xi_b2.normal_()
                            xi_b2.data = 1/(torch.abs(xi_b2.data) + self.tol)
                            xi = self.netGXi(
                                torch.cat([xi_b1, xi_b2], dim=1)
                                    ).view(self.batch_size, -1)
                        else:
                            xi_b.normal_()
                            xi = self.netGXi(xi_b).view(self.batch_size, -1)
                        x_fake = self.netG(z_b, xi)    
                    elif self.true_type == 'Gaussian':
                        x_fake = self.netG(z_b.normal_())
                    elif (self.true_type == 'Cauchy'):
                        z_b.normal_()
                        z_b.data.div_(torch.sqrt(
                            self.t_chi2_d.sample((self.batch_size, 1))
                                        ).to(self.device) + self.tol
                                     )
                        x_fake = self.netG(z_b)
                    feat_fake, g_fake_score = self.netD(x_fake)                    
                    if (floss == 'js'):
                        g_fake_loss = criterion(g_fake_score, one_b)
                    elif floss == 'tv':
                        g_fake_loss = - torch.sigmoid(g_fake_score).mean()
                    g_fake_loss.backward()
                    loss_G_ep.append(g_fake_loss.cpu().item())
                    self.optG.step()
                    if self.elliptical:
                        self.optGXi.step()
            ## Record intermediate error during training for monitoring.
            self.mean_err_record.append(
                (self.netG.bias.data - self.true_mean.to(self.device)).norm(2).item()
                )
            ## Record intermediate estimation during training for averaging.
            if (ep >= (epochs - avg_epochs)):
                self.mean_est_record.append(self.netG.bias.data.clone().cpu())
            self.loss_D.append(np.mean(loss_D_ep))
            self.loss_G.append(np.mean(loss_G_ep))
            ## Print intermediate result every verbose epoch.
            if ((ep+1) % verbose == 0):
                print('Epoch:%d, LossD/G:%.4f/%.4f, Error(Mean):%.4f' %
                    (ep+1, self.loss_D[-1], self.loss_G[-1], self.mean_err_record[-1]))
        ## Final results    
        self.mean_avg = sum(self.mean_est_record[-avg_epochs:])/\
                            len(self.mean_est_record[-avg_epochs:])
        self.mean_err_avg = (self.mean_avg - self.true_mean.cpu()).norm(2).item()
        self.mean_err_last = (
            self.netG.bias.data - self.true_mean.to(self.device)).norm(2).item()
        ## Print the final results.
        if show == True:
            self.netD.eval()
            ## Scores of true distribution from 10,000 samples.
            if self.true_type == 'Gaussian':
                t_x = self.t_d.sample((10000, ))
            elif self.true_type == 'Cauchy':
                t_normal_x = self.t_normal_d.sample((10000, ))
                t_chi2_x = self.t_chi2_d.sample((10000,))
                t_x = t_normal_x / (torch.sqrt(t_chi2_x.view(-1, 1)) + self.tol)
            self.true_D = self.netD(t_x.to(self.device))[1].detach().cpu().numpy()
            ## Scores of contamination distribution from 10,000 samples.
            if self.cont_type == 'Gaussian':
                c_x = self.c_d.sample((10000, )) + self.cont_mean.view(1, -1)
            elif self.cont_type == 'Cauchy':
                c_normal_x = self.c_normal_d.sample((10000, ))
                c_chi2_x = self.c_chi2_d.sample((10000,))
                c_x = c_normal_x / (torch.sqrt(c_chi2_x.view(-1, 1)) + self.tol) +\
                          self.cont_mean.view(1, -1)
            self.cont_D = self.netD(c_x.to(self.device))[1].detach().cpu().numpy()
            ## Scores of 10,000 generating samples.
            if self.elliptical:
                t_z = torch.randn(10000, self.p).to(self.device)
                t_z.div_(t_z.norm(2, dim=1).view(-1, 1) + self.tol)
                if use_inverse_gaussian:
                    t_xi1 = torch.randn(10000, self.g_input_dim//2).to(self.device)
                    t_xi2 = torch.randn(10000, self.g_input_dim//2).to(self.device)
                    t_xi2 = 1/(torch.abs(t_xi2.data) + self.tol)
                    xi = self.netGXi(
                        torch.cat([t_xi1, t_xi2], dim=1)).view(10000, -1)
                else:
                    t_xi = torch.randn(10000, self.g_input_dim).to(self.device)
                    xi = self.netGXi(t_xi).view(10000, -1)
                g_x = self.netG(t_z, xi).detach()
            elif self.true_type == 'Gaussian':
                g_x = self.netG(torch.randn(10000, self.p).to(self.device))
            elif (self.true_type == 'Cauchy'):
                g_z = torch.randn(10000, self.p).to(self.device)
                g_z.data.div_(
                    torch.sqrt(self.t_chi2_d.sample((10000, 1))
                                ).to(self.device) + self.tol
                             )
                g_x = self.netG(g_z)            
            self.gene_D = self.netD(g_x)[1].detach().cpu().numpy()
            ## Some useful prints and plots
            print('Avg error: %.4f, Last error: %.4f' % (self.mean_err_avg, self.mean_err_last))
            
            plt.plot(self.loss_D)
            plt.title('loss_D')
            plt.show()
            
            plt.plot(self.loss_G)
            plt.title('loss_G')
            plt.show()
            
            plt.plot(self.mean_err_record)
            plt.title('Error')
            plt.show()
            
            sns.distplot(self.true_D[(self.true_D < 25) & (self.true_D > -25)], hist=False, label='Dist of True')
            sns.distplot(self.gene_D[(self.gene_D < 25) & (self.gene_D > -25)], hist=False, label='Dist of Generated')
            sns.distplot(self.cont_D[(self.cont_D < 25) & (self.cont_D > -25)], hist=False, label='Dist of Contaminated')
            plt.legend()
            plt.title('Disc distribution')
            plt.show()

