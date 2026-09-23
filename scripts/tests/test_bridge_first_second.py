"""Input-connection regressions from the checked-in first-fold public values."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import bridge_first_second as bridge


class BridgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.first, self.second = self.root / "first", self.root / "second"
        self.generated = self.first / "step-1-to-2"
        self.original = self.second / "original-sources"
        tests = bridge.ROOT / "crates/nightstream/tests/fixtures"
        caller = bridge.load(tests / "lean/nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json")
        native = bridge.load(tests / "stage1_actual_nifs/actual_result.json")
        r = native["pi_rlc_parent"]
        phase = [0] * 10
        phase[0] = 1
        phase[3:8], phase[9] = r[:5], native["outgoing_state"]
        children = native["children"]
        phase_d = [0] * 17
        phase_d[0], phase_d[14], phase_d[16] = 1, phase[9], [1, children]
        nifs = [1, native["pi_ccs_input"], 0, 0, 0, native["pi_ccs_phase"],
                [0] * 6 + [native["package_identity"]], phase, 0, phase_d]
        wrapped = lambda word: {"value": word}
        extension = lambda value: {"value": list(map(wrapped, value)), "_phantom": None}
        padded = lambda values: list(map(extension, values)) + [extension([0, 0]) for _ in range(10)]
        frame = list(b"".join(word.to_bytes(8, "little") for word in caller[4][1]))
        parent = {"m_in": 270, "adv": None, "fold_digest": frame,
                  "c": {"d": 54, "kappa": 22, "data": list(map(wrapped, r[0]))},
                  "X": {"rows": 54, "cols": 5, "constant_hint": None, "packed_signed_unit": None,
                        "data": [wrapped(r[1][column * 54 + row]) for row in range(54) for column in range(5)]},
                  "r": list(map(extension, r[2])), "eval_k": padded(r[3]),
                  "eval_a": list(map(padded, r[4]))}
        request = [caller[2][28], caller[2][30:34], caller[2][35:39], caller[2][-4:]]
        next_request = [2, request[1], caller[4][0], request[3]]
        envelope = {"schema": 1, "child_witness_count": 16, "iteration": 2,
                    "z0": next_request[1], "current": next_request[2], "running_parent": parent,
                    "running_claims": [copy.deepcopy(parent) for _ in range(16)],
                    **{name: "metadata" for name in bridge.REFERENCE_METADATA}}
        public = [[], [], []]
        self.values = {
            self.generated / "caller.json": caller,
            self.generated / "nifs-result.json": nifs,
            self.generated / "children.json": children,
            self.generated / "next-message-input.json": next_request,
            self.first / "original-sources/next-message-input.json": request,
            self.original / "next-message-input.json": copy.deepcopy(next_request),
            self.original / "envelope.json": envelope,
            self.second / "step-2-to-3/caller.json": [1, caller[1]],
            self.second / "step-2-to-3/sources/public.json": public,
            self.second / "step-2-to-3/ccs-input.json": [1, public[0], public[1], 0, 0, 0, public[2]],
        }
        self.enterContext(patch.object(bridge, "load", side_effect=self.values.__getitem__))
        self.enterContext(patch.object(bridge.projection, "fresh_claim", return_value=([], caller[4][2])))

    def check_state(self):
        return bridge.state_and_parent(self.first, self.second)

    def test_recorded_first_state_matches_retained_input(self):
        self.assertEqual(self.check_state()["carried_frames"], 17)

    def test_changed_parent_fails_even_when_source_projection_is_unchanged(self):
        self.values[self.original / "envelope.json"]["running_parent"]["eval_k"][0]["value"][0]["value"] ^= 1
        with self.assertRaisesRegex(ValueError, "complete carried R parent"):
            self.check_state()

    def test_changed_last_child_digest_frame_fails(self):
        self.values[self.original / "envelope.json"]["running_claims"][15]["fold_digest"][31] ^= 1
        with self.assertRaisesRegex(ValueError, "successor frame"):
            self.check_state()

    def test_changed_next_message_fails(self):
        self.values[self.original / "next-message-input.json"][3][0] ^= 1
        with self.assertRaisesRegex(ValueError, "complete retained second request"):
            self.check_state()

    def test_retained_ccs_must_use_the_compared_public_input(self):
        self.values[self.second / "step-2-to-3/ccs-input.json"][6] = [1]
        with self.assertRaisesRegex(ValueError, "exact public input consumed by retained C"):
            self.check_state()

    def test_changed_complete_source_stream_fails_without_a_success_receipt(self):
        directory = self.root / "comparison"
        targets = (directory / "first-original", directory / "feedback", directory / "original",
                   self.generated / "sources", self.second / "step-2-to-3/sources")
        for target in targets:
            target.mkdir(parents=True)
            (target / "public.json").write_text("[]\n")
            (target / "sources.jsonl").write_text("[1]\n")
        (directory / "feedback/sources.jsonl").write_text("[2]\n")
        with self.assertRaisesRegex(ValueError, "complete first-successor/retained-input projection"):
            bridge.check(self.first, self.second, directory)
        self.assertFalse((directory / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
