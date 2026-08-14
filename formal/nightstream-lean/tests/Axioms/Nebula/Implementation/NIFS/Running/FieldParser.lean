import Nightstream.Implementation.Nebula.NIFS.Running.FieldParser
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ProductNifsFieldParser

/-- info: 'Nightstream.Implementation.Nebula.ProductNifsFieldParser.parse_rejects_modulus_word' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parse_rejects_modulus_word

/-- info: 'Nightstream.Implementation.Nebula.ProductNifsFieldParser.fieldWord_encode' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fieldWord_encode

/-- info: 'Nightstream.Implementation.Nebula.ProductNifsFieldParser.parse_encode' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parse_encode
