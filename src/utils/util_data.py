
# List of all synuclein names, using the specific syntax rules.
ALL_CONCEIVABLE_SYN_NAMES = ['asyn', 'bsyn', 'b3V', 'b4V', 'b5V', 'b5V2Q', 'b5V2G', 'b5V4Q', 'b5V4G', 'b5V6Q',
                             'b5V8Q', 'b8Q', 'aA30P', 'aE46K', 'aA53T', 'aT72V', 'aA76E', 'aE83Q', 'aS87E', 'aS129E',
                             'aA90P', 'aF94P', 'aF94V', 'aK45E', 'aK45V', 'aE46V', 'aK45VE46V', 'aV71GT72G',
                             'aV71ET72E', 'aK45VE46VV71ET72E', 'bR45V', 'bE46V', 'bR45VE46V', 'bV70M', 'bP123H',
                             'ba1', 'ba2', 'ba3', 'ba4', 'ba5', 'ba6', 'ba7', 'ba8', 'ba9', 'bM10K', 'ba11', 'ba12',
                             'ba13', 'ba14', 'ba15', 'ba16', 'ba17', 'fr_asyn', 'fr_bsyn', 'fr_gsyn1', 'fr_gsyn2',
                             'mus_bsyn', 'gallus_bsyn', 'a11_140', 'a21_140', 'a31_140', 'a41_140', 'a51_140',
                             'a61_140', 'a71_140', 'a1_45', 'a1_50', 'a1_55', 'a1_60', 'a1_70', 'a1_75', 'a1_80',
                             'gsyn', 'g1_80', 'ga', 'b1_73', 'a68_71del', 'a71_72del', 'a71_74del', 'a71_76del',
                             'a71_78del', 'a71_81del', 'a71_82del', 'a73_83del',  'a74_84del', 'a73_82del']


def check_syn_names(syn_names: list):
    for syn_name in syn_names:
        if syn_name not in ALL_CONCEIVABLE_SYN_NAMES:
            raise ValueError(f"'{syn_name}' not in list of synuclein names.")