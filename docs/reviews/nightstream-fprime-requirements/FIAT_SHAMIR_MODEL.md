# Classical Fiat–Shamir assumption boundary (retired)

**Retired on 2026-10-07 (owner decision).** The black-box transfer to the
causal interactive game has no useful `g`: guessing which oracle query carries
each SumCheck message loses about `(Q+1)^29`. `FiatShamirModel` and every
declaration that only it used are deleted.

The owner approved this assumption boundary on 2026-09-11 UTC. The approved
proposal had SHA-256
`207b67740a6948633870891048091a0c7b2dd90599c04392622ab5cc52cb7498`. Git
history holds the full record of the retired boundary.

The current model is the one-fold random-oracle theorem and, for the history,
HyperNova errata Assumption 1. See adversaries 3 and 4 and trust items 1 and 4
of [SECURITY_MODEL.md](../../../formal/nightstream-fprime/SECURITY_MODEL.md).
