import time
import os
import pandas as pd
import numpy as np
from data.protein_sequences import read_seqs
from src.salsa import Options
from src.salsa.Options import Props
from src.salsa import execute
from  root_path import abspath_root

start = time.time()
# STEP 0 - Which proteins are you interested in?
# accession_numbers = ['']
# accession_numbers = ['P37840']
# accession_numbers = ['P37840', 'Q16143', 'P10636-8']
protein_names = ['']
# protein_names = ['SYUA_HUMAN', 'SYUB_HUMAN']
# protein_names = ['TADBP_HUMAN']
# accession_numbers = ['P10636-5', 'P10636-8']
accession_numbers = ['P05067(672-711)', 'P05067(672-713)']
# accession_numbers = ['P10636-2', 'P10636-4', 'P10636-5']
# protein_names = ['SYUA_HUMAN', 'PRIO_HUMAN', 'URE2_YEAST', 'E9P8Q3_YEASX', 'TADBP_HUMAN']
prot_ids = protein_names + accession_numbers
prot_id_seqs = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids)

# STEP 1 - Define property and corresponding parameters.
_property = Props.mHA.value
# params = Options.DefaultMpp2HA.all_params.value
params = Options.DefaultMaHA.all_params.value


# STEP 2 - salsa produces an array holding a single numbers for each residue.
all_summed_scores = dict()
for prot_id, prot_seq in prot_id_seqs.items():
    scored_windows_all = execute.compute(sequence=prot_seq, _property=_property, params=params)
    summed_scores = execute.sum_scores_for_plot(scored_windows_all)
    all_summed_scores[prot_id] = summed_scores


# STEP 3 - Plot SALSA summed scores
# execute.plot_summed_scores(all_summed_scores, _property, prot_name_labels=list(all_summed_scores.keys()), params=params)

# STEP 4 - Generate a single scalar representing the property of interest for the protein of interest.
salsa_integrals = execute.integrate_salsa_plot(all_summed_scores)

# STEP 5 - Write summed scores, integrals & parameters to one csv file, per protein.
# Note the 4th column is params which has a different shape to the other 3 columns.
# This might therefore be a better match to a json format, however I have opted for csv files
# because csv files use about half the memory of json files.
# seq_aa|seq_nos|mHA|
params_ = dict()
params_['prot_id'] = None
params_['prop'] = _property
params_.update(params)
prop_dirname = '_'.join(_property.split(' '))

for (prot_id, aa_seq), (prot_id_, summed_scores), (prot_id__, salsa_integrals) in \
        zip(prot_id_seqs.items(), all_summed_scores.items(), salsa_integrals.items()):
    assert(prot_id == prot_id_ == prot_id__)
    params_['prot_id'] = prot_id
    aa_num = np.array(list(range(1, len(aa_seq) + 1))).transpose()
    aa_seq = np.array(list(aa_seq))
    summed_scores_for_csv = pd.DataFrame({'aa_seq': aa_seq, 'seq_num': aa_num, 'summed_scores': summed_scores})
    summed_scores_for_csv['params'] = np.nan
    summed_scores_for_csv['integral'] = np.nan
    summed_scores_for_csv['params'].iat[0] = params_
    summed_scores_for_csv['integral'].iat[0] = salsa_integrals
    timestamp = str(round(time.time()))
    path = os.path.join(abspath_root, 'data', 'salsa_outputs', prot_id, prop_dirname, 'summed_scores')
    try: os.makedirs(path)
    except OSError: print('Creation of directories %s failed' % path)
    csv_ = os.path.join(path, timestamp + '.csv')
    summed_scores_for_csv.to_csv(csv_)

print(f'{round(1000 * (time.time() - start), 1)} ms')
