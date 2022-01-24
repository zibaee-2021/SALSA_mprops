import time
from src.steps.SALSA import salsa_prot_props
from src.salsa_calculator import compute_salsa as salsa

start = time.time()
# STEP 0 - Which proteins are you interested in?
accesssion_numbers = ['']
protein_name = ['']

# STEP 1 - What property are you interested in?
_property = salsa_prot_props.SalsaProtProps.SALSA_bSC.value

# STEP 2 - SALSA produces an array holding a single numbers for each residue.
all_summed_scores = salsa.run(accs=accesssion_numbers, prot_names=protein_name, _property=_property)

# STEP 3
# currently only able to plot one protein per plot.
salsa.plot_summed_salsa(all_summed_scores, _property, protein_name)

# STEP 4 - Generate a single scalar representing the property of interest for the protein of interest.
salsa_integrals = salsa.integrate_salsa_plot(all_summed_scores)

print(f'{round(1000 * (time.time() - start), 1)} ms')