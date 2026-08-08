# neo-fold-clean

`neo-fold-clean` owns Nightstream's lifecycle and protocol integration.

| Module | Role |
|---|---|
| `lifecycle/` | Public preprocess, prove, extend, finish, and verify flow |
| `paper/nifs/` | PiCCS, PiRLC, and PiDEC composition |
| `paper/construction2/` | Recursive state transition and terminal fold |
| `paper/f_prime/` | Native and constrained F' semantics |
| `frontends/f_prime/` | Shared low-norm F' image |
| `frontends/r1cs_f_prime/` | Authoritative recursive R1CS relation |
| `frontends/nebula/` | Nebula memory relation and recursive lifecycle |
| `frontends/direct_ccs/` | Caller-supplied CCS audit path |
| `engine/decider.rs` | Full-history audit R1CS |
| `frontends/r1cs_f_prime/terminal_r1cs/` | Terminal R1CS and WIP Spartan bridge |

The recursive R1CS and Nebula frontends prove the F' induction. Direct CCS
does not, so its multi-chunk verification path replays the audit trail.

See [Lifecycle](../architecture/lifecycle.md),
[Frontends](../architecture/frontends.md), and
[Terminal proof](../architecture/decider.md).
