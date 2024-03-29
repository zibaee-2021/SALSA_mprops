import time
from AA import AA
from src.mutation import mutator
from data import write_outputs
from src.salsa.Options import Props, DefaultMLSC
from src.salsa import salsa

if __name__ == '__main__':
    start = time.time()

    # STEP 0 - SELECT PROTEINS & MUTATIONS:
    point_mutants = {'P05067(672-713)': {1: [AA.Alanine.value], 5: [AA.Cysteine.value],
                                         20: [AA.Histidine.value, AA.Valine.value]}}
    prot_ids_seqs_mutant_ids_seqs = mutator.make_point_mutants(point_mutants)
    prot_id_seqs = prot_ids_seqs_mutant_ids_seqs

    # STEP 1 - SELECT PROPERTY AND CORRESPONDING PARAMETERS:
    _property = Props.mLSC.value
    params = {'window_len_min': DefaultMLSC.window_len_min.value,
              'window_len_max': DefaultMLSC.window_len_max.value,
              'top_scoring_windows_num': DefaultMLSC.top_scoring_windows_num.value,
              'threshold': DefaultMLSC.threshold.value,
              'abs_threshold': DefaultMLSC.abs_threshold.value}

    # STEP 2 - RUN SALSA TO GENERATE SUMMED SCORES PER RESIDUE:
    all_summed_scores = dict()
    for prot_id, mutant_ids_seqs in prot_ids_seqs_mutant_ids_seqs.items():
        for mutant_id, mutant_seq in mutant_ids_seqs.items():
            scored_windows_all = salsa.compute_all_scored_windows(sequence=mutant_seq, _property=_property, params=params)
            summed_scores = salsa.sum_scores_for_plot(scored_windows_all)
            all_summed_scores[mutant_id] = summed_scores

    # STEP 3 - PLOT OUTPUT OF PREVIOUS STEP:
    salsa.plot_summed_scores(all_summed_scores, _property, prot_name_labels=list(all_summed_scores.keys()),
                               params=params)

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