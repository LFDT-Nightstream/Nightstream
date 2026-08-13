import Nightstream.Checks.Rust

/-! Regression for fail-closed Rust conformance status output. -/

namespace NightstreamTests.RustCheckStatus

#guard Nightstream.Checks.Rust.resultLine true ==
  "rust_model_checks=pass; M5=bounded-rust-origin-gate-required; direct_terminal_spartan=artifact-checked-bounded-lockstep; DEC-SOUND=model-proved-reduction; production-transfer=open"

#guard Nightstream.Checks.Rust.resultLine false ==
  "rust_model_checks=fail; M5=fail; DEC-SOUND production-transfer=open"

#guard !Nightstream.Checks.Rust.containsSubstr
  (Nightstream.Checks.Rust.resultLine false) "M5-pass"

#guard !Nightstream.Checks.Rust.containsSubstr
  (Nightstream.Checks.Rust.resultLine true) "M5-pass"

end NightstreamTests.RustCheckStatus
