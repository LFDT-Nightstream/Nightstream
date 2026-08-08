import Nightstream.Checks.Rust

/-! Regression for fail-closed Rust conformance status output. -/

namespace NightstreamTests.RustCheckStatus

#guard Nightstream.Checks.Rust.resultLine true ==
  "rust_conformance=M5-reopened (functional probes and artifact checks pass; Rust-originated provenance audit open); direct_terminal_spartan=artifact-checked-bounded-lockstep; generic_compact_decider=not-exposed; DEC-SOUND=open"

#guard Nightstream.Checks.Rust.resultLine false ==
  "rust_conformance=M5-fail; no Rust-conformant claim is established; DEC-SOUND=open"

#guard !Nightstream.Checks.Rust.containsSubstr
  (Nightstream.Checks.Rust.resultLine false) "M5-pass"

#guard !Nightstream.Checks.Rust.containsSubstr
  (Nightstream.Checks.Rust.resultLine true) "M5-pass"

end NightstreamTests.RustCheckStatus
