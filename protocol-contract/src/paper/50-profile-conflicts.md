## PAPER-PROFILE — Appendix B.2 Goldilocks profile

`PAPER-PROFILE-001` The paper reports:

```text
q = 2^64-2^32+1              K = F_(q^2)
eta = 81                     Phi = X^54+X^27+1
d = 54                       kappa = 18
n_F = 2^30                   b = 2
k = 14                       K_fresh <= 61
B = 2^14                     T = 216
C coefficients = {-2,-1,0,1,2}
abs(C) about 2^125           MSIS estimate about 129 bits.
```

Source: `SRC-PAPER-11`, lines 13-19.

## PAPER-CONFLICT-DIM — incompatible printed dimensions

`PAPER-CONFLICT-001` The reviewed paper still requires `n_F=d*n_R`, but its
Goldilocks profile selects `d=54` and `n_F=2^30`. Since
`2^30 mod 54=46`, these values do not define an integer `n_R`. The joint PiCCS
also uses one power-of-two domain with `m=n_F`.

This extraction does not resolve the conflict. Nightstream decisions
`NSD-DOMAIN-001` and `NSD-DOMAIN-MAP-001` instead select the actual logical
widths and one larger padded row cube. They do not treat the printed
`n_F=2^30` value as a valid ring-aligned committed-vector width.

Source: `SRC-PAPER-04`, `SRC-PAPER-07`, and `SRC-PAPER-11`.

## PAPER-CONFLICT-LENGTH — overloaded commitment length

`PAPER-CONFLICT-002` Definition 4 declares `Setup(1^lambda,m)` and
`Commit(pp,z)` for `z` in `R_F^m`. Definition 14 and the reduction Setup
algorithms instead call `Setup(1^lambda,n_R)` and use
`L:R_F^(n_R)->C`. The two length parameters have the same name and different
meanings.

This extraction reads the commitment length as `n_R`, the committed-vector ring width.
A concrete Module-SIS claim must state which length it uses.

Source: `SRC-PAPER-04` and `SRC-PAPER-07`.
