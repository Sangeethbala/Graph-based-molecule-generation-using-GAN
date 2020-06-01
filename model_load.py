import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch

import pandas as pd
from mmltoolkit.featurizations import Estate_CDS_SoB_featurizer
from rdkit import Chem
import pickle

from jtvae.fast_jtnn.mol_tree import Vocab, MolTree
from jtvae.fast_jtnn.jtnn_vae import JTNNVAE

import matplotlib.pyplot as plt

import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})
from scipy.stats import uniform
import numpy as np


from sklearn.neighbors import KernelDensity
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# import math
import sys
import pdb
sys.stdin = open('/dev/tty')
pdb.set_trace()

cc = [.60, 0.6958140728520726, 0.6316306226176055, 0.538368249782626, 0.5564464627171499, 0.6578173393570554, 0.6155282697237867, 0.6800328576941783, 0.661955028973716, 0.70812472544951, 0.6792788120243978, 0.6545856607352715, 0.6749984960700484, 0.6218542086173372, 0.6229637104280574]


def verify_sequence(smile):
    """Returns True if the SMILES string is valid and
    its length is less than max_len."""
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1        
        
def batch_det_vel(smiles, train_smiles=None, det_vel_model=None, dens_model=None, logh50_model=None, bond_names=None):
    """
    Prediction of detonation velocity (and other things) with a scikit-learn model
    """
    
    """
    Loads the a scikit-learn model for det_vel and parameters for featurization (bond_names)
    """

    det_vel_model = pickle.load(open('/home/sbalakri/mol-cycle-gan/wgan_vae/model_KRR_Estate_CDS_SoB_det_V_425_mols.pkl', 'rb'))
    dens_model = pickle.load(open('/home/sbalakri/mol-cycle-gan/wgan_vae/model_KRR_Estate_CDS_SoB_density_425_mols.pkl', 'rb'))
    logh50_model = pickle.load(open('/home/sbalakri/mol-cycle-gan/wgan_vae/model_KRR_Estate_CDS_SoB_logh50_mattheiu.pkl', 'rb'))
    bond_names = pickle.load(open('/home/sbalakri/mol-cycle-gan/wgan_vae/bond_names.pkl', 'rb'))
    
    
    
    if det_vel_model == None:
        raise ValueError('The det vel model was not properly loaded.')
    if bond_names == None:
        raise ValueError('The bond type names for the det vel prediction were not properly loaded.')

    fsmiles = []
    zeroindex = []
    num_valid = 0
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
            num_valid = num_valid+1
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)

    fmols = [Chem.MolFromSmiles(smiles) for smiles in fsmiles]

    X_Estate_CDS_SoB = Estate_CDS_SoB_featurizer(fmols, predefined_bond_types=bond_names, scaled=False, return_names=False)

    # reweight to lie between ~ [0,1]
    vals = np.asarray(det_vel_model.predict(X_Estate_CDS_SoB))/11.0

    # include sensitivity prediction, and reweight things to lie roughly between ~ [0,1]
    if not (logh50_model == None):
        vals = vals/2.0 + (1 - np.asarray(logh50_model.predict(X_Estate_CDS_SoB))/3.0)/2.0

    for k in zeroindex:
        vals[k] = 0.0

    vals = np.squeeze(vals)
    return vals, num_valid

def reward_func(enc_vec, train_file):
    

    if train_file:
        rewards = batch_det_vel(enc_vec)
    else:
        returned_smiles = []
        tree_dims = int(56 / 2)
        mols = enc_vec.detach()
    #     mols.detach().numpy()
        for i in range(mols.shape[0]):

    #         tree_vec = np.expand_dims(mols[i, 0:tree_dims], 0)
    #         mol_vec = np.expand_dims(mols[i, tree_dims:], 0)
            tree_vec = mols[i, 0:tree_dims]
            mol_vec = mols[i, tree_dims:]
            tree_vec = torch.reshape(tree_vec, (1, 28))
            mol_vec = torch.reshape(mol_vec, (1, 28))
            tree_vec = torch.autograd.Variable(tree_vec.float())
            mol_vec = torch.autograd.Variable(mol_vec.float())
            smi = model.decode(tree_vec, mol_vec, prob_decode=False)
            returned_smiles.append(smi)
            print(i)
#             if i==10: break


        rewards, num_valid = batch_det_vel(returned_smiles)

    return rewards, num_valid

def load_model(opts):
    vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depth = int(opts.depth)
 
    model = JTNNVAE(vocab, hidden_size, latent_size, 20, depth)
    model.load_state_dict(torch.load(opts.model_path))

    return model.cuda()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
#         self.model = nn.Sequential(
#              nn.Linear(5, 16),
#              nn.LeakyReLU(0.2, inplace=True),
#              nn.Linear(16, 32),
#              nn.LeakyReLU(0.2, inplace=True),
#              nn.Linear(32, 56),
#              nn.Tanh()
#          )
        self.model = nn.Sequential(
            *block(5, 8, normalize=True),
            *block(8, 16),
            *block(16, 32),
            nn.Linear(32, 56),
            nn.Tanh()
        )


    def forward(self, z):

        img = self.model(z)
        img = img.view(img.shape[0], 56)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(56, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class Options:
    def __init__(self,
                 jtvae_path="./jtvae/",
                 hidden_size=450,
                 latent_size=56,
                 depth=3,
                 jtnn_model_path="fast_molvae/vae_model/model.iter-3600",
                 vocab_path="data/energetics/vocab.txt"):
        self.jtvae_path = jtvae_path
        self.vocab_path = os.path.join(jtvae_path, vocab_path)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.model_path = os.path.join(jtvae_path, jtnn_model_path)
    
  
parser = argparse.ArgumentParser()    
parser.add_argument("--jtvae_path", default="/home/sbalakri/mol-cycle-gan/jtvae/")
parser.add_argument("--jtnn_model_path", default="fast_molvae/vae_model/model.iter-3600")
parser.add_argument("--vocab_path", default="data/energetics/vocab.txt")

parser.add_argument("--hidden_size", default=450, type=int)
parser.add_argument("--latent_size", default=56, type=int)
parser.add_argument("--depth", default=3, type=int)

opt = parser.parse_args()
print(opt)

jtvae_path_tuple = (opt.jtvae_path, opt.jtnn_model_path, opt.vocab_path)
jtvae_setting_tuple = (opt.hidden_size, opt.latent_size, opt.depth)

jtvae_path, jtnn_model_path, vocab_path = jtvae_path_tuple
hidden_size, latent_size, depth = jtvae_setting_tuple

opts = Options(jtvae_path=jtvae_path,
               hidden_size=hidden_size,
               latent_size=latent_size,
               depth=depth,
               jtnn_model_path=jtnn_model_path,
               vocab_path=vocab_path)
model = load_model(opts)    
    
    
find_mean = False
if find_mean:
#     import sys
#     import pdb
#     sys.stdin = open('/dev/tty')
#     pdb.set_trace()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()

    train_file = False
    latent_dim = 5
    mean_rew = []
    for i in range(96):
        if i % 5 == 0:
            filepath = '/home/sbalakri/mol-cycle-gan/2norm_generator_'+str(i)+'.t7'
            generator.load_state_dict(torch.load(filepath))

            filepath = '/home/sbalakri/mol-cycle-gan/2norm_discriminator_'+str(i)+'.t7'

            discriminator.load_state_dict(torch.load(filepath))

            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

            z = Variable(Tensor(np.random.normal(0, 1, (580, latent_dim))))

            gen_imgs = generator(z)
            
            from sklearn.externals import joblib
    
            scaler = joblib.load('scaler.save') 
            gen_imgs = gen_imgs.cpu().detach().numpy()
            gen_imgs = scaler.inverse_transform(gen_imgs)
            gen_imgs = torch.from_numpy(gen_imgs).cuda().float()

            reward = reward_func(gen_imgs, train_file)

            mean_reward = np.mean(reward)
            mean_rew.append(mean_reward)
            print(mean_rew)
    
    import sys
    import pdb
    sys.stdin = open('/dev/tty')
    pdb.set_trace()
    
    
    plt.hist(mean_rew, bins = 40)
    plt.show()
train_file = False  
if train_file:

    data_path = '/home/sbalakri/mol-cycle-gan/data/results/aromatic_rings/energetics_1.txt'
    smiles_df = pd.read_csv(data_path)
    smiles_df = smiles_df.iloc[:, 0]
    
    reward = reward_func(smiles_df, train_file)
    
    import sys
    import pdb
    sys.stdin = open('/dev/tty')
    pdb.set_trace()
    ax = sns.distplot(reward, bins=20, kde=True,color='black', hist_kws={"linewidth": 15,'alpha':1}, norm_hist=True)
    ax.set(xlabel='Reward ', ylabel='Frequency')
    plt.show()
    plt.hist(reward, bins = 20)
    plt.show()
    
else:

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()

    filepath = '/home/sbalakri/mol-cycle-gan/4norm_generator_48.t7'

    generator.load_state_dict(torch.load(filepath))
#     model.eval()

    filepath = '/home/sbalakri/mol-cycle-gan/4norm_discriminator_48.t7'

    discriminator.load_state_dict(torch.load(filepath))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    latent_dim = 5
    z = Variable(Tensor(np.random.normal(0, 1, (580, latent_dim))))

    gen_imgs = generator(z)
    from sklearn.externals import joblib
    
    scaler = joblib.load('scaler.save') 
    import sys
    import pdb
    sys.stdin = open('/dev/tty')
    pdb.set_trace()
    gen_imgs = gen_imgs.cpu().detach().numpy()
    gen_imgs = scaler.inverse_transform(gen_imgs)
    gen_imgs = torch.from_numpy(gen_imgs).cuda().float()
    
    reward, num_valid = reward_func(gen_imgs, train_file)
    
    import sys
    import pdb
    sys.stdin = open('/dev/tty')
    pdb.set_trace()
    print(num_valid)
    

    
    
    ax = sns.distplot(reward, bins=20, kde=True,color='black', hist_kws={"linewidth": 15,'alpha':1})
    ax.set(xlabel='Reward ', ylabel='Frequency')
    plt.show()
    
    plt.hist(reward, bins = 20)
    plt.show()