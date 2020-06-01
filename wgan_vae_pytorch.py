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

import torch.nn as nn
import torch.nn.functional as F
import torch

import pandas as pd
from mmltoolkit.featurizations import Estate_CDS_SoB_featurizer
from rdkit import Chem
import pickle

from jtvae.fast_jtnn.mol_tree import Vocab, MolTree
from jtvae.fast_jtnn.jtnn_vae import JTNNVAE

import sys
import pdb
sys.stdin = open('/dev/tty')
pdb.set_trace()

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
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
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
    return vals

def reward_func(enc_vec):
    
    returned_smiles = []
    tree_dims = int(56 / 2)
    mols = enc_vec.detach()
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

    rewards = batch_det_vel(returned_smiles)

    return rewards

def load_model(opts):
    vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depth = int(opts.depth)
 
    model = JTNNVAE(vocab, hidden_size, latent_size, 20, depth)
    model.load_state_dict(torch.load(opts.model_path))

    return model.cuda()

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=5, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=2, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")


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

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=False))
            return layers
#         self.model = nn.Sequential(
#              nn.Linear(opt.latent_dim, 16),
#              nn.LeakyReLU(0.2, inplace=True),
#              nn.Linear(16, 32),
#              nn.LeakyReLU(0.2, inplace=True),
#              nn.Linear(32, 56),
#              nn.Tanh()
#          )
        self.model = nn.Sequential(
            *block(opt.latent_dim, 8, normalize=True),
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
            nn.ReLU(inplace=False),
            nn.Linear(32, 16),
            nn.ReLU(inplace=False),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
 
# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
data_path = "/home/sbalakri/mol-cycle-gan/data/results/aromatic_rings/encoded_list_.csv"
smiles_df = pd.read_csv(data_path, index_col=0)
mols = smiles_df.values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(mols)
mols1 = scaler.transform(mols)
# import sys
# import pdb
# sys.stdin = open('/dev/tty')
# pdb.set_trace()

from sklearn.externals import joblib
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename) 

dataloader = torch.utils.data.DataLoader(
    mols1,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, imgs in enumerate(dataloader):
       
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
#         print(imgs)
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)
   
            
        # Train the generator every n_critic iterations
#         if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z)


        reward = reward_func(gen_imgs)

        # Adversarial loss
        loss_G = -(torch.mean(discriminator(gen_imgs) + .5 * torch.from_numpy(reward).cuda().float()))

        loss_G.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
        )
        
    PATH = '/home/sbalakri/mol-cycle-gan/5norm_generator_'+str(epoch)+'.t7'
    torch.save(generator.state_dict(), PATH)

    PATH = '/home/sbalakri/mol-cycle-gan/5norm_discriminator_'+str(epoch)+'.t7'
    torch.save(discriminator.state_dict(), PATH)

#         if batches_done % opt.sample_interval == 0:
#             save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
#         batches_done += 1