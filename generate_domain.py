import os
import numpy as np
import pandas as pd
import argparse
import torch
from src.encoding_utils import generate_onehot, generate_all_combos

"""
Script to generate a domain (all of the variants and encodings to enumerate across) for the combinatorial library, for a production (wet-lab) run. Only needs to be run once before an ALDE campaign.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ParPgb")
    parser.add_argument("--nsites", type=int, default=5)
    args = parser.parse_args()
    
    #set the number of residues in the combinatorial library and the name for the campaign
    nsites = args.nsites
    name = args.name

    #set path for to store results
    path = 'data/' + name + '/'
    #make the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    #generate strings for all combos in the design space
    all_combos = generate_all_combos(nsites)
    df = pd.DataFrame(all_combos, columns=['Combo'])
    df.to_csv(path + 'all_combos.csv', index=False)

    #generate onehot encoding for all combos
    X = torch.reshape(generate_onehot(all_combos), (len(all_combos), -1))
    torch.save(X, path + 'onehot_x.pt')

    #implement your own code to generate other desired encodings