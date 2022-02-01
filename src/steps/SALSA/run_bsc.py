import time
from data.protein_sequences import read_seqs
from src.salsa.Options import Props, DefaultBSC
from src.salsa import execute

start = time.time()
# STEP 0 - Which proteins are you interested in?
# accession_numbers = ['']
# accession_numbers = ['P37840']
# accession_numbers = ['P37840', 'Q16143', 'P10636-8']
accession_numbers = ['P10636-8']
# accession_numbers = ['P05067(672-713)', 'P05067(672-711)']
# accession_numbers = ['P10636-5', 'P10636-8']
# protein_names = ['PRIO_HUMAN']
# protein_names = ['SYUA_HUMAN']
# protein_names = ['TADBP_HUMAN']
# protein_names = ['SYUA_HUMAN', 'PRIO_HUMAN', 'URE2_YEAST', 'E9P8Q3_YEASX', 'TADBP_HUMAN']
protein_names = ['']
prot_ids = accession_numbers + protein_names
prot_id_seqs = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids=prot_ids)

# STEP 1 - Define property and corresponding parameters.
_property = Props.bSC.value
params = DefaultBSC.all_params.value

# STEP 2 - salsa produces an array holding a single numbers for each residue.
all_summed_scores = dict()
for prot_id, prot_seq in prot_id_seqs.items():
    scored_windows_all = execute.compute(sequence=prot_seq, _property=_property, params=params)
    summed_scores = execute.sum_scores_for_plot(scored_windows_all)
    all_summed_scores[prot_id] = summed_scores

# STEP 3 - Plot SALSA summed scores
execute.plot_summed_scores(all_summed_scores, _property, prot_name_labels=list(all_summed_scores.keys()), params=params)

# STEP 4 - Generate a single scalar representing the property of interest for the protein of interest.
salsa_integrals = execute.integrate_salsa_plot(all_summed_scores)

# STEP 5 - Write out all_summed_scores and salsa integrals
# TODO

print(f'{round(1000 * (time.time() - start), 1)} ms')