"""
Module to apply a sampling strategy for detecting the optimal configuration of side-chain-side-chain
interactions in a stacked amyloid fold.
The theory is that by scoring expected side-chain-side-chain interactions according to relative
levels of complementarity (or otherwise), the sum of all these values for each possible configuration
will correlate with observed data. Such a finding would strongly suggest that the population of
proteins sample every possible arrangement and that those which are observed to form are the most
stable on average for all the residues in the amyloid core.

A further demonstration will be to compare the results of using the whole protein sequence versus
just the region(s) identified by high beta-strand contiguity.

This would provide evidence that the formation of the filament might be a combination of beta-stand
contiguity causing one b-sheet face to start to grow, before the chains collapse into the folds
that further stabilise and provide a template for 'true' seeding, leading to exact replicas of the
origin seed(s).
The way in which every possible conformation of polypeptide in the grid/graph representation would
be calculated from is shown here using a very small 3-residue peptide. It should be possible to see
that residue `1` is fixed at the central position, while residues `2` and `3` sample every possible
conformation with respect to `1`. Blank grids indicate where the intended conformation is not possible
due co-location of two residues for the intended conformation.
In this sequential representation, it can be seen that each residue samples every neighbouring position
of its N-terminal neighbour in a clockwise order and including all 8 possible positions, i.e. north,
north-east, east, south-east, south, south-west, west, north-west:
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - 2 3 - -   - - 2 3 -   - - - 2 3
- - 1 2 3   - - 1 - -   - - 1 - -   - - 1 - -   - - - - -   - - 1 - -   - - 1 - -   - - 1 - -
- - - - -   - - - 2 3   - - 2 3 -   - 2 3 - -   - - - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -

- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - 2 - -   - - - 2 -
- - 1 2 -   - - 1 - -   - - 1 - -   - - 1 - -   - 2 1 - -   - - - - -   - - 1 3 -   - - 1 - 3
- - - - 3   - - - 2 -   - - 2 - -   - 2 - - -   - - 3 - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - 3   - - - 3 -   - - 3 - -   - - - - -   - - - - -   - - - - -   - - - - -

- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - 2 - - -   - - - - -   - - - 2 -
- - 1 2 -   - - 1 - -   - - 1 - -   - - 1 - -   - 2 1 - -   - 3 1 - -   - - - - -   - - 1 3 -
- - - 3 -   - - - 2 -   - - 2 - -   - 2 - - -   - 3 - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - 3 -   - - 3 - -   - 3 - - -   - - - - -   - - - - -   - - - - -   - - - - -
etc..


Sampling scores the potential complementarity of each conformation according to the properties
of its neighbouring residues.
The above can be implemented by a graph that simply serves as the implementation of the grid above,
but I think it also as likely that a fully connected graph could be another, if not a better, graph
model to go about this task. I will need to investigate this after first implementing the original
idea.
"""


