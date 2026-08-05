# Reviewed SuperNeo errata v4

Status: **applied to the reviewed paper snapshot**.

The v4 patch contains the reviewed v3 security corrections and later
post-review source corrections. Each row below has one semantic meaning. A
general "source cleanup" row cannot close one of these obligations.

| ID | Corrected rule | Current source |
|---|---|---|
| ERR-DEGREE-SEMANTICS | `D_f` is the total degree of `f`; `D_f<=u` is a separate structure invariant. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:5-17` |
| ERR-DIM-DEGREE | A valid Structure owns the exact matrix dimensions, `D_f`, and the `D_f<=u` proof obligation. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:5-17` |
| ERR-SUMCHECK-DEGREE | The joint polynomial has maximum individual degree `D_Q=max(D_f+1,2b,2)`. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:71-82` |
| ERR-CCS-ZERO-SET | The CCS constraint polynomial must vanish on the Boolean cube. It is not an integer-set membership claim. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:18-25` |
| ERR-LIN-DOMAIN | `L_in` has the full committed-vector ring domain and projects the input prefix. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:26-29` |
| ERR-CE-TYPES | The CE witness has field width `n_F`; calls to `L` and `L_in` use the stated coefficient embedding. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:26-34` |
| ERR-NORM-ROOTS | Use `P_b(Z)=product_(a=-(b-1))^(b-1)(Z-iota_q(a))`, the exact strict centered range test. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:68-82` |
| ERR-ABS-TARGET | Use `T_abs=gamma^(2K+k)T_local`; fresh, norm, and carried exponent blocks must be disjoint. | `protocol-contract/paper-sources/13-d-deferred-theorems-and-proofs.md:105-117` |
| ERR-AMBIENT | Use `B_amb=floor(q/2)+1` only for the universal ambient relation. Keep `B=b^k` for protocol and binding assumptions. | `protocol-contract/paper-sources/04-4-preliminaries.md:17-21,29-37` |
| ERR-COORD-FORK | Charge the exact PiRLC coordinate-fork loss `(K+k)/abs(C)`, or the reviewed conservative `(K+k+1)/abs(C)` corollary. | `protocol-contract/paper-sources/13-d-deferred-theorems-and-proofs.md:362-374` |
| ERR-PIRLC-PROJECTION | PiRLC extraction uses the corrected projection and ambient relation. | `protocol-contract/paper-sources/13-d-deferred-theorems-and-proofs.md:330-430` |
| ERR-STRONG-EXTRACT | Run one unconditioned PiCCS execution, enter the retry loop only after success, and charge the global `sqrt(delta)` disagreement loss. | `protocol-contract/paper-sources/13-d-deferred-theorems-and-proofs.md:179-330` |
| ERR-EXTRACT-RUNTIME | The retry-runtime expectation is conditioned on entry into the success-gated loop before it is averaged over inputs. | `protocol-contract/paper-sources/13-d-deferred-theorems-and-proofs.md:205-235` |
| ERR-ERROR-BUDGET | SumCheck false-claim acceptance and gamma-batching Schwartz--Zippel failure are separate terms. | `protocol-contract/paper-sources/09-supplementary-material.md:11-17` |
| ERR-PIDEC-EQUATIONS | PiDEC checks commitment and evaluation recomposition; completeness derives public-input recomposition from the verifier-computed split. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:151-176` |
| ERR-EVALUATION-NOTATION | CE and PiDEC use the transformed matrix-vector multilinear evaluation consistently. | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md:26-34,151-176` |

The machine-readable erratum-to-rule mapping is derived from normative rule
citations and appears in `rule-index.json`. This table does not own that edge.

## Other post-review corrections

Errata v4 also makes the mixed-radix map `I` explicitly bijective, corrects
the Appendix B.3 extension-field type, corrects the introductory challenge
field wording, and repairs the Boolean-cube statement in Lemma 6. These changes
are source-locked. They do not resolve the dimension defect below.

## Remaining paper defect

Definition 1 requires `n_F=d*n_R`. Appendix B.2 selects `d=54` and
`n_F=2^30`, but `2^30 mod 54=46`. Joint PiCCS also assumes one power-of-two
domain with `m=n_F`. Errata v4 does not resolve this defect. Nightstream v1
selects its actual ring-aligned assignment width, its actual row count, and one
larger zero-padded row cube. This is a Nightstream decision, not paper errata.
