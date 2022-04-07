import time
from data.protein_sequences import read_seqs
from src.salsa.Options import Props, DefaultMEDM
from src.salsa import salsa
from data import write_outputs

start = time.time()

# STEP 0 - SELECT PROTEINS
protein_names, accession_numbers = [''], ['']
# accession_numbers = ['P10636-5', 'P10636-8']
accession_numbers = ['P10636-2', 'P10636-4', 'P10636-5']
# accession_numbers = ['P10636-2', 'P10636-4', 'P10636-5', 'P10636-6', 'P10636-7', 'P10636-8']
# accession_numbers = ['P37840']
# accession_numbers = ['P37840', 'Q16143', 'P10636-8']
# protein_names = ['SYUA_HUMAN']
# protein_names = ['TADBP_HUMAN']
# protein_names = ['SYUA_HUMAN', 'PRIO_HUMAN', 'URE2_YEAST', 'E9P8Q3_YEASX', 'TADBP_HUMAN']

prot_ids = accession_numbers + protein_names
prot_id_seqs = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids)

# STEP 1 - SELECT PROPERTY AND CORRESPONDING PARAMETERS:
_property = Props.mEDM.value
# params = {'window_len_min': DefaultMEDM.window_len_min.value,
#           'window_len_max': DefaultMEDM.window_len_max.value,
#           'top_scoring_windows_num': DefaultMEDM.top_scoring_windows_num.value,
#           'threshold': DefaultMEDM.threshold.value,
#           'abs_threshold': DefaultMEDM.abs_threshold.value,
#           'periodicity': DefaultMEDM.periodicity.value}
params = DefaultMEDM.all_params.value

# STEP 2 - RUN SALSA TO GENERATE SUMMED SCORES PER RESIDUE:
all_summed_scores = dict()
for prot_id, prot_seq in prot_id_seqs.items():
    scored_windows_all = salsa.compute_all_scored_windows(sequence=prot_seq, _property=_property, params=params)
    summed_scores = salsa.sum_scores_for_plot(scored_windows_all)
    all_summed_scores[prot_id] = summed_scores

# STEP 3 - PLOT OUTPUT OF PREVIOUS STEP:
salsa.plot_summed_scores(all_summed_scores, _property, prot_name_labels=list(all_summed_scores.keys()), params=params)

# STEP 4 - SUM SCORES TO SCALAR:
salsa_integrals = salsa.integrate_salsa_plot(all_summed_scores)

# STEP 5 - WRITE RESULTS TO CSV:
params_ = dict()
params_['prot_id'] = None
params_['prop'] = _property
params_.update(params)
prop_dirname = '_'.join(_property.split(' '))
write_outputs.write_csv(params_, prop_dirname, prot_id_seqs, all_summed_scores, salsa_integrals)

print(f'{round(1000 * (time.time() - start), 1)} ms')