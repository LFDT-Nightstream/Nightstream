# neo-ccs

CCS structures and claims that the reductions use.

- `CcsStructure`: the matrices `M_j` and the polynomial `f`.
- `Mat` (dense, with compact signed-unit and constant storage), and `CscMat` and `CcsMatrix` (sparse).
- `CcsClaim`, `CcsWitness` and `CeClaim`: the fresh and evaluation claims that PiCCS, PiRLC and PiDEC use.
- `crypto::poseidon2_goldilocks`: the Poseidon2 permutation and sponge of the protocol.
- SuperNeo matrix transforms, and conversion from R1CS to CCS.
