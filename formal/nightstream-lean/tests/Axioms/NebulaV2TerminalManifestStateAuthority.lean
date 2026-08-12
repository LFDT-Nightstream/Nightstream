import Nightstream.Implementation.NebulaV2.TerminalManifestStateAuthority
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall.Call.priorStateCarryPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestNifsCall.Call.priorStateCarryPlaced

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestStateAuthority.incomingAuthority_digest_eq_columns' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestStateAuthority.incomingAuthority_digest_eq_columns

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestStateAuthority.boundaryFromPrevious' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestStateAuthority.boundaryFromPrevious
