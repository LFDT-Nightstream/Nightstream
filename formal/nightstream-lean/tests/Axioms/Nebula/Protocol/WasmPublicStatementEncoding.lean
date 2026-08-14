import Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
import tests.Axioms.Support

/-! Dependency audit for public completion arithmetic. -/

open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding

/-- info: 'Nightstream.Protocol.Nebula.WasmPublicStatementEncoding.PublicImage.Decodes.completionTrace' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PublicImage.Decodes.completionTrace

/-- info: 'Nightstream.Protocol.Nebula.WasmPublicStatementEncoding.PublicImage.DecodesFor.completionTrace' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PublicImage.DecodesFor.completionTrace
