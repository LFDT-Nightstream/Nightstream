import Nightstream.Implementation.NebulaV2.Commitment.Compact.TokenRows
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.CompactTokenRows.four_setup_seeds_pairwise_distinct' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CompactTokenRows.four_setup_seeds_pairwise_distinct

/-- info: 'Nightstream.Implementation.NebulaV2.CompactTokenRows.token_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactTokenRows.token_exact
