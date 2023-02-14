import time
from data.protein_sequences import read_seqs
from data import write_outputs
from src.salsa.Options import Props, DefaultBSC
from src.salsa import salsa

if __name__ == '__main__':
    start = time.time()

    # STEP 0 - SELECT PROTEINS:
    accession_numbers, protein_names = [''], ['']
    # accession_numbers = ['']
    # accession_numbers = ['P37840']
    # accession_numbers = ['P37840', 'Q16143', 'P10636-8']
    # accession_numbers = ['P05067(672-713)', 'P05067(672-711)']
    # accession_numbers = ['P10636-5', 'P10636-8']
    # protein_names = ['PRIO_HUMAN']
    protein_names = ['SYUA_HUMAN', 'SYUB_HUMAN']
    # protein_names = ['TADBP_HUMAN']
    # protein_names = ['SYUA_HUMAN', 'PRIO_HUMAN', 'URE2_YEAST', 'E9P8Q3_YEASX', 'TADBP_HUMAN']
    # protein_names = ['SYUA_FUGU', 'SYUB_FUGU', 'SYUG1_FUGU', 'SYUG2_FUGU']
    # protein_names =['T106B_HUMAN', 'SYUA_HUMAN']
    prot_ids = accession_numbers + protein_names
    prot_id_seqs = read_seqs.get_seqs_by_uniprot_acc_nums_or_names(prot_ids=prot_ids)

    # STEP 1 - SELECT PROPERTY AND CORRESPONDING PARAMETERS:
    _property = Props.bSC.value
    params = DefaultBSC.all_params.value

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
    params_['prot_id'] = None  # Init as placeholder, assigned in write_outputs().
    params_['prop'] = _property
    params_.update(params)
    prop_dirname = '_'.join(_property.split(' '))
    write_outputs.write_csv(params_, prop_dirname, prot_id_seqs, all_summed_scores, salsa_integrals)

    print(f'{round(1000 * (time.time() - start), 1)} ms')