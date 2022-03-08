import os
import numpy as np
import pandas as pd
import time
from root_path import abspath_root


def write_csv(params_, prop_dirname, prot_id_seqs, all_summed_scores, salsa_integrals):
    for (prot_id, aa_seq), (prot_id_, summed_scores), (prot_id__, salsa_integrals) in \
            zip(prot_id_seqs.items(), all_summed_scores.items(), salsa_integrals.items()):
        assert (prot_id == prot_id_ == prot_id__)
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