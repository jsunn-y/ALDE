import numpy as np
import torch
from src.encoding_utils import generate_onehot, generate_all_combos

"""
Script to generate a domain for the combinatorial library. Only needs to be run once before an active learning campaign
"""
#set the number of residues in the combinatorial library and the name for the campaign
nsites = 5
name = 'ParPgb'

#set path for to store results
path = 'data/' + name + '/'

#generate strings for all combos in the design space
all_combos = generate_all_combos(nsites)
np.save(path + "combos.npy", np.array(all_combos))

#generate onehot encoding for all combos
X = torch.reshape(generate_onehot(all_combos), (len(all_combos), -1))
torch.save(X, path + 'onehot_x.pt')

#implement your own code to generate other desired encodings