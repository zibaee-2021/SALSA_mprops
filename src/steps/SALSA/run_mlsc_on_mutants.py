import time
from AA import AA
from src.mutation import mutator
from src.salsa.Options import Props, DefaultMLSC
from src.salsa import execute

start = time.time()

# STEP 0 - Which proteins and amino acid substitution mutants are you interested in?
point_mutants = {'P05067(672-713)': {1: [AA.Alanine.value], 5: [AA.Cysteine.value],
                                     20: [AA.Histidine.value, AA.Valine.value]}}

# STEP 1 - Make the mutations you want.
prot_ids_seqs_mutant_ids_seqs = mutate.make_point_mutants(point_mutants)

# STEP 2 - Define parameters for SALSA low-sequence-complexity on the sequences.
_property = Props.mLSC.value
params = {'window_len_min': DefaultMLSC.window_len_min.value,
          'window_len_max': DefaultMLSC.window_len_max.value,
          'top_scoring_windows_num': DefaultMLSC.top_scoring_windows_num.value,
          'threshold': DefaultMLSC.threshold.value,
          'abs_threshold': DefaultMLSC.abs_threshold.value}

# STEP 3 - Run SALSA
all_summed_scores = dict()
for prot_id, mutant_ids_seqs in prot_ids_seqs_mutant_ids_seqs.items():
    for mutant_id, mutant_seq in mutant_ids_seqs.items():
        scored_windows_all = execute.compute(sequence=mutant_seq, _property=_property, params=params)
        summed_scores = execute.sum_scores_for_plot(scored_windows_all)
        all_summed_scores[mutant_id] = summed_scores

    # STEP 4 - Plot SALSA summed scores
    execute.plot_summed_scores(all_summed_scores, _property, prot_name_labels=list(all_summed_scores.keys()),
                               params=params)

    # STEP 5 - Sum scores to a scalar
    salsa_integrals = execute.integrate_salsa_plot(all_summed_scores)

    # STEP 6 - Write out all_summed_scores and salsa integrals
    # TODO

print(f'{round(1000 * (time.time() - start), 1)} ms')