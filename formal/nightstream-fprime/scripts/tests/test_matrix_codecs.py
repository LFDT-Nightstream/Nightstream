from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
import check_matrix_codecs as codecs


VALID = '''import NightstreamFPrime.Layout.MatrixProgram
import NightstreamFPrime.Export.Codec
namespace NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Export.Codec

def Thing.format : Format Thing where
  encode := fun thing => .array [.atom thing.value]
  decode
    | .array [.atom value] => .ok ⟨value⟩
    | _ => .error "invalid thing"
  decode_encode := by
    intro thing
    cases thing
    rfl

end NightstreamFPrime.Layout.MatrixProgram
'''


class MatrixCodecTests(unittest.TestCase):
    def test_format_fields_and_decode_encode_proof_are_allowed(self):
        self.assertEqual(codecs.check_source(VALID), 1)

    def test_multiline_format_type_is_allowed(self):
        source = VALID.replace('Thing.format : Format Thing where',
                               'Thing.format :\n    Format (Thing) where')
        self.assertEqual(codecs.check_source(source), 1)

    def test_physical_definition_is_rejected(self):
        for prefix in ['', 'private ', '@[inline] private ']:
            with self.subTest(prefix=prefix):
                source = VALID + prefix + 'def rowCount : Nat := 4\n'
                with self.assertRaisesRegex(ValueError, 'Format-typed'):
                    codecs.check_source(source)

    def test_physical_structure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'physical declaration'):
            codecs.check_source(VALID + 'structure PhysicalRow where\n  column : Nat\n')

    def test_indented_physical_definition_cannot_hide_in_codec_body(self):
        source = VALID.replace('    rfl', '    rfl\n  def rowCount : Nat := 4')
        with self.assertRaisesRegex(ValueError, 'Format-typed'):
            codecs.check_source(source)

    def test_comments_and_strings_cannot_create_declarations(self):
        source = '/- outer /- structure Fake where -/ def hidden : Nat := 0 -/\n' + VALID
        source = source.replace('"invalid thing"',
            '"structure Fake where\ndef rowCount : Nat := 0"')
        source += '-- def another : Nat := 0\n'
        self.assertEqual(codecs.check_source(source), 1)

    def test_comments_and_strings_cannot_supply_format_type(self):
        for decoy in ['/- : Format Thing where -/', '"Format Thing where"']:
            with self.subTest(decoy=decoy):
                with self.assertRaisesRegex(ValueError, 'Format-typed'):
                    codecs.check_source(VALID + f'def wrong : Nat := {decoy} 0\n')

    def test_format_word_does_not_allow_a_different_result_type(self):
        for result in ['Nat', 'Format Thing × Nat', 'Format Thing → Nat']:
            with self.subTest(result=result):
                source = VALID.replace('Format Thing where', result + ' where')
                with self.assertRaisesRegex(ValueError, 'Format-typed'):
                    codecs.check_source(source)

    def test_unrecognized_module_command_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsupported command'):
            codecs.check_source(VALID + 'initialize hidden : Nat ← pure 0\n')

    def test_only_matrix_program_export_paths_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            export = project / 'NightstreamFPrime/Export'
            nested = export / 'MatrixProgram/Nested'
            nested.mkdir(parents=True)
            (export / 'MatrixProgram.lean').write_text(VALID)
            (export / 'Other.lean').write_text('structure AllowedElsewhere where\n  value : Nat\n')
            self.assertEqual(codecs.check_project(project), (1, 1))
            (nested / 'Physical.lean').write_text('def rowCount : Nat := 4\n')
            with self.assertRaisesRegex(ValueError, 'Format-typed'):
                codecs.check_project(project)


if __name__ == '__main__':
    unittest.main()
