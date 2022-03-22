import time
from data.protein_sequences import read_seqs
from src.salsa import Options
from src.salsa.Options import Props
from src.salsa import salsa
from src.salsa import similarity

start = time.time()
# STEP 0 - Which proteins are you interested in?
# accession_numbers = ['']
# accession_numbers = ['P37840']
# accession_numbers = ['P37840', 'Q16143', 'P10636-8']
# protein_names = ['']
base_prot_id = 'P10636-5'
query_prot_id = 'P10636-8'
protein_names = [base_prot_id, query_prot_id]
# protein_names = ['TADBP_HUMAN']
# accession_numbers = ['P10636-5', 'P10636-8']
# accession_numbers = ['P10636-2', 'P10636-4', 'P10636-5']
# protein_names = ['SYUA_HUMAN', 'PRIO_HUMAN', 'URE2_YEAST', 'E9P8Q3_YEASX', 'TADBP_HUMAN']
accession_numbers = ['']
prot_ids = protein_names + accession_numbers
prot_id_seqs = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids)

# STEP 1 - Define property and corresponding parameters.
_property = Props.bSC.value
params = Options.DefaultBSC.all_params.value

# STEP 2 - salsa produces an array holding a single numbers for each residue.
all_summed_scores = dict()
for prot_id, prot_seq in prot_id_seqs.items():
    scored_windows_all = salsa.compute_all_scored_windows(sequence=prot_seq, _property=_property, params=params)
    summed_scores = salsa.sum_scores_for_plot(scored_windows_all)
    all_summed_scores[prot_id] = summed_scores

# STEP 3 - Plot SALSA summed scores
# execute.plot_summed_scores(all_summed_scores, _property, prot_name_labels=list(all_summed_scores.keys()), params=params)

# STEP 4 - Generate a single scalar representing the property of interest for the protein of interest.
salsa_integrals = salsa.integrate_salsa_plot(all_summed_scores)

# STEP 5 - Run similarity scan & plot result
similarity_scores = similarity.compute_sum_of_products_of_summed_scores(base_summed_scores=all_summed_scores[base_prot_id],
                                                                        query_summed_scores=all_summed_scores[query_prot_id],
                                                                        query_window_len_min=2,
                                                                        query_window_len_max=10)

prots = ' and '.join(list(all_summed_scores.keys()))
prots_similarity_scores = {prots: similarity_scores}
prot_name_labels = ['Similarity of ' + prots + _property]
salsa.plot_summed_scores(prots_similarity_scores, _property=_property + ' similarity',
                           prot_name_labels=prot_name_labels, params=params)

# STEP x - Write out all_summed_scores and salsa integrals
# TODO

print(f'{round(1000 * (time.time() - start), 1)} ms')