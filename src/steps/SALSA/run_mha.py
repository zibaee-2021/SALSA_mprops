import time
import os
import pandas as pd
import numpy as np
from data.protein_sequences import read_seqs
from data import write_outputs
from src.salsa import Options
from src.salsa.Options import Props
from src.salsa import salsa


start = time.time()
# STEP 0 - Which proteins are you interested in?
accession_numbers, protein_names = [''], ['']
# accession_numbers = ['P37840']
# accession_numbers = ['P37840', 'Q16143', 'P10636-8']
# protein_names = ['SYUA_HUMAN', 'SYUB_HUMAN']
# protein_names = ['TADBP_HUMAN']
# accession_numbers = ['P10636-5', 'P10636-8']
# accession_numbers = ['P05067(672-711)', 'P05067(672-713)']
protein_names = ['T106B_HUMAN']
# accession_numbers = ['P10636-2', 'P10636-4', 'P10636-5']
# protein_names = ['SYUA_HUMAN', 'PRIO_HUMAN', 'URE2_YEAST', 'E9P8Q3_YEASX', 'TADBP_HUMAN']
prot_ids = protein_names + accession_numbers
prot_id_seqs = read_seqs.get_seqs_by_uniprot_acc_nums_or_names(prot_ids)

# STEP 1 - Define property and corresponding parameters.
_property = Props.mHA.value
# params = Options.DefaultMpp2HA.all_params.value
params = Options.DefaultMaHA.all_params.value


# STEP 2 - salsa produces an array holding a single numbers for each residue.
all_summed_scores = dict()
for prot_id, prot_seq in prot_id_seqs.items():
    scored_windows_all = salsa.compute_all_scored_windows(sequence=prot_seq, _property=_property, params=params)
    summed_scores = salsa.sum_scores_for_plot(scored_windows_all)
    all_summed_scores[prot_id] = summed_scores


# STEP 3 - Plot SALSA summed scores
# salsa.plot_summed_scores(all_summed_scores, _property, prot_name_labels=list(all_summed_scores.keys()), params=params)

# STEP 4 - Generate a single scalar representing the property of interest for the protein of interest.
salsa_integrals = salsa.integrate_salsa_plot(all_summed_scores)

# STEP 5 - Write summed scores, integrals & parameters to one csv file, per protein.
# Note the 4th column is params which has a different shape to the other 3 columns.
# As such, this seems a better match to a json format, however I have opted for csv files
# because csv files use about half the memory of json files.
# seq_aa|seq_nos|mHA|
params_ = dict()
params_['prot_id'] = None
params_['prop'] = _property
params_.update(params)
prop_dirname = '_'.join(_property.split(' '))
write_outputs.write_csv(params_, prop_dirname, prot_id_seqs, all_summed_scores, salsa_integrals)

print(f'{round(1000 * (time.time() - start), 1)} ms')
