import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Statement

/-! Axiom audits for the shape of the public statement of the replay
(`Replay/Statement.lean`): the profile headers of the schedule rows, the consequences of
`Public.shaped`, and the axes of the points. `Statement.exists_shaped` is the guard that
`Statement.Shaped` has an instance for each shape, and `tests/AkitaAllVectors.lean` runs
`Public.shaped` with `root20` and `root30` on the statements of four real openings. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.littleEndian_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tableDigest_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.MatrixProfile.bytes_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Profile.bytes_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.profiles20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.profiles30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.root20_vars
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.root30_vars
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rootGroup_header_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.root20_rows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.root30_rows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rootGroups_getElem?
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.axes20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.axes30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.sealed_shaped
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.sealed_layout
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.sealed_plan
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.sealed_groups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.sealed_root
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.shaped_word
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Statement.exists_shaped
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.shaped_groups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.shaped_group
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.shaped_values
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Statement.Shaped.rows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.shaped20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.shaped30
