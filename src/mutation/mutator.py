from typing import List
from enum import Enum
from data.protein_sequences import read_seqs


_20_AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def _are_valid_positions_to_mutate(prot_id_seqs: dict, prot_id_point_mutations_to_make: dict[str: List[dict]]) -> bool:
    """
    Check whether the requested mutation is compatible with the protein sequence. i.e. Mutation to any residues at
    position 150 for protein that is 140 residues long is not possible.
    :param prot_id_seqs: Protein ids/names mapped to protein sequence. Protein sequence should be in 1-letter notation.
    :param prot_id_point_mutations_to_make: Protein ids/names mapped to amino acid substitution to make.
    Example of expected format is {'SYUA_HUMAN': [{30: 'P'}], 'P10636-7': [{301: 'S'}, {301: 'L'}]}.
    :return: True if all the requested mutations are possible.
    """
    for prot_id, seq in prot_id_seqs.items():
        requested_mutations = prot_id_point_mutations_to_make[prot_id]
        if isinstance(requested_mutations, dict):
            requested_mutations = [requested_mutations]
        for requested_mutation in requested_mutations:
            position, = requested_mutation.keys()
            len_ = len(seq)
            if 1 <= position <= len(seq):
                continue
            else:
                print(f"You can't mutate {prot_id} at {position}. "
                      f'1 <= valid positions <= {len(seq)}')
                return False
    return True


def _make_mutant_name(prot_id_seq: dict[str: str], point_mutation_to_make: dict[int: str]) -> str:
    """
    Add suffix to given protein id to indicate the amino acid substitution.
    :param prot_id_seq: Protein id mapped to its sequence, {'SYUA_HUMAN': 'MDVFMKGLS..'}. Protein sequence should be
    in 1-letter notation.
    :param point_mutation_to_make: Position mapped to amino acid to mutate into. {1: 'A'}
    :return: The given protein id/name with suffix indicating the amino acid substitution,
    e.g. {'SYUA_HUMAN': 'MDVFMKGLS..'} and {1: 'A'} returns 'SYUA_HUMAN(M1A)'
    """
    (prot_id, seq), = prot_id_seq.items()
    position, = point_mutation_to_make.keys()
    original_aa = seq[position - 1]
    new_aa, = point_mutation_to_make.values()
    return f'{prot_id}({original_aa}{position}{new_aa})'


def _make_point_mutants(prot_id_seq: dict[str: str], point_mutants_to_make: dict[int: List[str]]) -> dict[str:str]:
    """
    Generate new id suffix and mutated sequence for the given sequence according to the specified amino acid
    substitution. Include original protein name/id and its wild-type sequence.
    :param prot_id_seq: Protein id/name mapped to its sequence. (Protein sequence should be in 1-letter notation.)
    e.g. {'A4_HUMAN(672-713)' :'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'}
    :param point_mutants_to_make: Amino acid substitution, e.g. {1: ['Y', 'T'], 2 ['A', 'C']}
    :return: Name of new mutant mapped to the mutated sequence, as well as wild-type.
    e.g. {'A4_HUMAN(672-713)' :'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
          'A4_HUMAN(672-713)(D1Y)' :'YAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
          'A4_HUMAN(672-713)(D1T)' :'TAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'}}
    """
    mutant_names_seqs = {}
    (prot_id, seq), = prot_id_seq.items()
    mutant_names_seqs[prot_id] = seq

    for position in list(point_mutants_to_make.keys()):
        for mutant_aa in point_mutants_to_make[position]:
            seq_ = list(seq)
            seq_[position - 1] = mutant_aa
            mutant_seq = ''.join(seq_)
            mutant_name = _make_mutant_name(prot_id_seq, {position: mutant_aa})
            mutant_names_seqs[mutant_name] = mutant_seq
    return mutant_names_seqs


def make_point_mutants(prot_id_mutants_to_make: dict[str: dict[int: List[str]]]) -> dict[str: dict[str:str]]:
    """
    Generate collection of sequences with specified amino acid substitutions for the given protein.
    (Protein sequences in 1-letter notation.)
    :param prot_id_mutants_to_make: Protein ids/names mapped to a collection of position-amino acid pairings.
    E.g. {'A4_HUMAN(672-713)': {1: ['Y', 'F']},
                   'P10636-7': {301: ['S', 'L']}}
    :return: Protein ids each mapped to corresponding mutated sequences. Protein sequence in 1-letter notation.
    e.g. {'A4_HUMAN(672-713)':
                            {'A4_HUMAN(672-713)' :'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
                            'A4_HUMAN(672-713)(D1Y)' :'YAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
                            'A4_HUMAN(672-713)(D1F)' :'FAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'},
        'P10636-7':
                    {'P10636-7(P301S): 'MAEPRQEFEVMEDH...',
                    {'P10636-7(P301L): 'MAEPRQEFEVMEDH...',}}
    """
    prot_ids_mutant_ids_seqs = dict()
    for prot_id, pos_residues in prot_id_mutants_to_make.items():
        prot_id_seq = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids=prot_id)
        mutant_ids_seqs = _make_point_mutants(prot_id_seq=prot_id_seq, point_mutants_to_make=pos_residues)
        prot_ids_mutant_ids_seqs[prot_id] = mutant_ids_seqs
    return prot_ids_mutant_ids_seqs


def make_fragment(syn_name: str) -> str:
    prot = ''
    asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_HUMAN')['SYUA_HUMAN']
    bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_HUMAN')['SYUB_HUMAN']
    gsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG_HUMAN')['SYUG_HUMAN']
    if syn_name[0] == 'a':
        prot = asyn
    elif syn_name[0] == 'b':
        prot = bsyn
    elif syn_name[0] == 'g':
        prot = gsyn
    else:
        print(f'Character should be a, b or g. Character passed was {syn_name[0]}')
    if 'Del' in syn_name:
        start_end = syn_name[1:-3].split('_')
        fragment = prot[: int(start_end[0]) - 1] + prot[int(start_end[1]):]
    else:
        start_end = syn_name[1:].split('_')
        fragment = prot[int(start_end[0]) - 1: int(start_end[1])]
    return fragment


def mutate(prot_seq: str, pos_aa: dict) -> str:
    """
    Mutate given protein sequence at the given position(s) to the given residue(s).
    e.g. You want to mutate your sequence: ACDEFG to ACDYFG. Here pos_aa should be {4: 'Y'}. This will replace the 'E'
    at array index position 3 to 'Y'.
    :param prot_seq: Protein sequence in 1-letter notation.
    :param pos_aa: The position(s) in the sequence (using 1 to indicate position 0 in a zero-based array) to mutate
    mapped to the residue(s) to mutate to. E.g. To mutate a given protein sequence at positions 4, 30 and 101 to
    Alanine, Glutamine and Tyrosine, respectively, the argument is {4: 'A', 30: 'Q', 101: 'Y'}.
    :return: The mutated protein sequence. (Sequence in 1-letter notation).
    """
    mutated_seq = ''
    for pos, aa in pos_aa.items():
        if not 1 <= pos <= len(prot_seq):
            print(f"Error: select position between 1 and {len(prot_seq)}"
                  f"\nYou specified position: {pos}")
        elif aa not in _20_AA:
            print(f"Error: select one of 20 amino acids: {_20_AA} "
                  f"\nYou specified unrecognised character: '{aa}'")
        else:
            mutated_seq = prot_seq[:pos - 1] + aa + prot_seq[pos:]
        prot_seq = mutated_seq
    return mutated_seq


def mutate_all_sequence(mut_option):
    """
    :param mut_option: One of three following options can be chosen:
    1. to perform specific amino acid substitutions at specific positions in the given protein(s);
    2. to perform specific amino acid substitutions at every positions in the given protein(s);
    3. to perform all 20 amino acid substitutions at every position in the given protein(s).
    :return:
    """
    # TODO
    pass


class Mutate(Enum):
    GivenAAsAtGivenPositions = 1
    All20AAsAtGivenPositions = 2
    All20AAsAtAllPossiblePositions = 2


    # @classmethod
    # def default(cls):
    #     return cls.GivenAAsAtGivenPositions


if __name__ == '__main__':

    actual = make_point_mutants(prot_id_mutants_to_make={'P37840': {1: ['A'], 3: ['D']}})
    # mydict = {'prot': {1: ['A'], 3: ['D', 'E']}, 'prot2': {1: ['A']}}
    # blaa, blaaa = mydict.keys()
    # for key in mydict.keys():
    #     print(key)
    # blaa, blaaa = mydict.items()

    # print(blaa)
    # print(blaaa)

    # for name, pos_mut in bla.items():
    #     blbla, = pos_mut.keys()

    # mydict = {'prot': {1: ['A' 'C']}}
    # bla, = mydict.keys()
    # mydict = {'prot': {1: ['A'], 3: ['D', 'E']}}
    # bla, = mydict.keys()

