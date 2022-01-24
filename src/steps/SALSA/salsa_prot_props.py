from enum import Enum, unique


@unique
class SalsaProtProps(Enum):

    SALSA_bSC = 'beta_strand_contiguity'
    SALSA_mHA = 'mean_helical_amphipathicity'
    SALSA_LSC = 'low-sequence-complexity'
