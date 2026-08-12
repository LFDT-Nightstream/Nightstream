import Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
import tests.Axioms.Support

/-! Dependency audit for public completion arithmetic. -/

open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding

/-- info: 'Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding.PublicImage.Decodes.completionTrace' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PublicImage.Decodes.completionTrace

/-- info: 'Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding.PublicImage.DecodesFor.completionTrace' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PublicImage.DecodesFor.completionTrace
