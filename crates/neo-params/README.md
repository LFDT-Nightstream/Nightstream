# neo-params

Typed parameter sets with validation and security estimates for Nightstream protocol.

## Surface
- `NeoParams` struct with typed presets for different security levels and field choices.
- **Appendix B.2 Goldilocks preset**: Goldilocks field, η=81, s=2, paper challenge set size |C|≈2^125, expansion T=216.
- **Guard enforcement**: validates `(k+1)·T·(b−1) < B` at parameter load; rejects unsafe combinations.
- **Extension degree computation**: computes minimal s for target soundness (v1 supports s=2 only).

## Presets
- **goldilocks_paper_b2**: SuperNeo Appendix B.2 parameters with Goldilocks prime
  - Base field: q = 2^64 - 2^32 + 1
  - Extension: K = F_{q^2} (s=2) 
  - Cyclotomic: η=81, d=φ(81)=54, Φ_η = X^54 + X^27 + 1
  - Ajtai: κ=18 rows, m=2^30 columns
  - Norm schedule: b=2, k=14, B=2^14=16384
  - Strong sampler: T=216 from coeffs ∈ [-2,-1,0,1,2]
  - **Verified**: (k+1)·T·(b−1) = 15·216·1 = 3240 < 16384 ✓

- **MERSENNE61_128**: Alternative using 2^61-1 prime (when supported)
- **TOY**: Insecure parameters for testing and development only

## Requirements
- **MUST** enforce parameter safety: reject any preset where (k+1)·T·(b−1) ≥ B
- **MUST** compute minimal extension degree s for target soundness λ (currently fixed s=2)
- **MUST** export all parameters as typed values: (q,η,d,κ,m,b,k,B,T,C,s)
- **MUST** provide security estimates and validation methods

## Tests
- Parameter consistency validation across all presets
- Guard inequality verification for Appendix B.2 Goldilocks and other secure presets
- Security estimate regression tests
