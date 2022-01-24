# Read me

---


Simple ALgorithm of Sliding Averages (SALSA) is published in Protein Science (2007) 16:906-918.


https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2206631/pdf/906.pdf

Figure 5 provides a clear explanation of the beta-strand contiguity SALSA computation.

Figure 6 provides a clear explanation of the alpha-helical amphipathicity SALSA computation.


The beta-strand contiguity and sliding averages idea was first conceived 2002-2003. 
It was originally written in Java and further developed and refined before publication in 2007. 

---


mprops

Early ideas that would eventually lead to mprops algorithm were also conceived in 2002-2003. 
It was developed and refined in close coordination with experimental data in vitro. 
It was published in Journal of Biological Chemistry (2010) 285(49): 38555-38567.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2992288/pdf/zbc38555.pdf

The end of 'Experimental Procedures' on page 38555 gives the calculation of mprops:

   mprops = a(nMP) - b(nMHL) - c(nMNC) - d(nMTC) 
   where n indicates a normalized value

Combining mprops & SALSA beta-SC:

   combined = e(nmprops) + f(nbeta-SC)

Based on the observed lag times for 32 synuclein constructs, the fitted values of the coefficients are:
a = 0.074, b = 0.066, c = 0.063, d = 0.060, e = 0.0435 and f = 0.0283 
(Rsquared = 0.71). For any given construct, using this model, the combined calculated properties thus give a 
prediction of relative fibrillogenic propensity.
