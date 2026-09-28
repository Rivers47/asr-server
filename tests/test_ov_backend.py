"""OpenVINO backend logic that does not need the model loaded.

ov_backend imports transformers defensively, so this module imports cleanly in the
CPU environment and these tests run there: the processor is faked and the decode
step is stubbed, leaving the span, window and token-budget arithmetic under test.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ov_backend import (  # noqa: E402
    HOTWORDS_TOKEN_CAP,
    MAX_NEW_TOKENS,
    SPECIAL_TOKEN_SLOTS,
    WHISPER_SAMPLING_RATE,
    WINDOW_SAMPLES,
    OpenVinoWhisperModel,
    RawSegment,
    _spans_from_clip_timestamps,
    prompt_from_hotwords,
)


class FakeProcessor:
    """get_prompt_ids the way WhisperProcessor does: <|startofprev|> then one id per character."""

    START_OF_PREV = 50362

    def get_prompt_ids(self, text, return_tensors=None):
        return [self.START_OF_PREV] + [1000 + index for index, _ in enumerate(text)]


class SpanTest(unittest.TestCase):
    def test_no_clip_timestamps_is_the_whole_file(self):
        self.assertEqual(_spans_from_clip_timestamps(None, 42.0), [(0.0, 42.0)])
        self.assertEqual(_spans_from_clip_timestamps([], 42.0), [(0.0, 42.0)])

    def test_flat_pairs(self):
        """The format vad_segments_to_clip_timestamps emits for the unbatched path."""
        self.assertEqual(_spans_from_clip_timestamps([1.0, 2.5, 4.0, 9.0], 20.0), [(1.0, 2.5), (4.0, 9.0)])

    def test_trailing_start_runs_to_the_end(self):
        self.assertEqual(_spans_from_clip_timestamps([1.0, 2.5, 7.0], 20.0), [(1.0, 2.5), (7.0, 20.0)])

    def test_comma_separated_string(self):
        self.assertEqual(_spans_from_clip_timestamps("1.0,2.5,4.0,9.0", 20.0), [(1.0, 2.5), (4.0, 9.0)])

    def test_batched_dict_form(self):
        clips = [{"start": 1.0, "end": 2.0}, {"start": 3.0, "end": 4.0}]
        self.assertEqual(_spans_from_clip_timestamps(clips, 20.0), [(1.0, 2.0), (3.0, 4.0)])

    def test_empty_spans_are_dropped(self):
        self.assertEqual(_spans_from_clip_timestamps([5.0, 5.0, 6.0, 8.0], 20.0), [(6.0, 8.0)])


class HotwordsPromptTest(unittest.TestCase):
    def test_no_hotwords_leaves_the_default_budget(self):
        prompt_ids, max_new_tokens = prompt_from_hotwords(FakeProcessor(), "", 448)
        self.assertIsNone(prompt_ids)
        self.assertEqual(max_new_tokens, MAX_NEW_TOKENS)

    def test_budget_accounts_for_prompt_and_special_tokens(self):
        """transformers raises when prompt + specials + max_new_tokens exceeds 448."""
        hotwords = "x" * 100  # 100 ids plus <|startofprev|>
        prompt_ids, max_new_tokens = prompt_from_hotwords(FakeProcessor(), hotwords, 448)
        self.assertEqual(len(prompt_ids), 101)
        self.assertEqual(max_new_tokens, 448 - 101 - SPECIAL_TOKEN_SLOTS)
        self.assertLessEqual(len(prompt_ids) + SPECIAL_TOKEN_SLOTS + max_new_tokens, 448)

    def test_short_hotwords_do_not_raise_the_default(self):
        _, max_new_tokens = prompt_from_hotwords(FakeProcessor(), "xy", 448)
        self.assertEqual(max_new_tokens, MAX_NEW_TOKENS)

    def test_long_hotwords_are_capped_like_faster_whisper(self):
        prompt_ids, max_new_tokens = prompt_from_hotwords(FakeProcessor(), "x" * 400, 448)
        self.assertEqual(len(prompt_ids), HOTWORDS_TOKEN_CAP + 1)
        self.assertGreaterEqual(max_new_tokens, 1)
        self.assertLessEqual(len(prompt_ids) + SPECIAL_TOKEN_SLOTS + max_new_tokens, 448)


class TranscribeAssemblyTest(unittest.TestCase):
    """transcribe() windowing and timestamp offsets, with the decode step stubbed."""

    def _model(self, decode_result=None):
        model = object.__new__(OpenVinoWhisperModel)  # no IR, no OpenVINO runtime
        model.model_dir = "models/ov-int8"
        model.device = "CPU"
        model.processor = FakeProcessor()
        model.max_target_positions = 448
        model._reported_ignored = set()
        self.windows: list[tuple[float, int]] = []

        def fake_decode(window, offset_s, params):
            self.windows.append((offset_s, len(window)))
            self.params = params
            return list(decode_result(window, offset_s) if decode_result else [])

        model._decode_window = fake_decode
        return model

    def test_one_window_per_30s_and_offsets_are_absolute(self):
        model = self._model()
        audio = np.zeros(WHISPER_SAMPLING_RATE * 70, dtype=np.float32)
        _, info = model.transcribe(audio, language="ja", task="transcribe", beam_size=1)
        self.assertEqual(info.duration, 70.0)
        self.assertEqual([offset for offset, _ in self.windows], [0.0, 30.0, 60.0])
        self.assertEqual(
            [length for _, length in self.windows], [WINDOW_SAMPLES, WINDOW_SAMPLES, WHISPER_SAMPLING_RATE * 10]
        )

    def test_clip_timestamps_restrict_and_offset_the_windows(self):
        model = self._model()
        audio = np.zeros(WHISPER_SAMPLING_RATE * 100, dtype=np.float32)
        model.transcribe(audio, clip_timestamps=[10.0, 20.0, 50.0, 95.0])
        self.assertEqual([offset for offset, _ in self.windows], [10.0, 50.0, 80.0])
        self.assertEqual(
            [length for _, length in self.windows],
            [WHISPER_SAMPLING_RATE * 10, WINDOW_SAMPLES, WHISPER_SAMPLING_RATE * 15],
        )

    def test_segments_keep_the_absolute_times_the_decoder_returned(self):
        model = self._model(decode_result=lambda window, offset: [RawSegment(offset + 1.0, offset + 2.0, "ちゅぷっ")])
        audio = np.zeros(WHISPER_SAMPLING_RATE * 45, dtype=np.float32)
        segments, _ = model.transcribe(audio)
        self.assertEqual([(s.start, s.end) for s in segments], [(1.0, 2.0), (31.0, 32.0)])

    def test_info_omits_duration_after_vad(self):
        """The pipeline falls back to its own speech total when the attribute is absent."""
        model = self._model()
        _, info = model.transcribe(np.zeros(WHISPER_SAMPLING_RATE * 5, dtype=np.float32))
        self.assertFalse(hasattr(info, "duration_after_vad"))

    def test_hotwords_reach_the_decoder_as_prompt_ids(self):
        model = self._model()
        model.transcribe(np.zeros(WHISPER_SAMPLING_RATE * 5, dtype=np.float32), hotwords="ちゅぷっ", beam_size=5)
        self.assertEqual(self.params["num_beams"], 5)
        self.assertIsNotNone(self.params["prompt_ids"])
        self.assertLess(self.params["max_new_tokens"], MAX_NEW_TOKENS)

    def test_ignored_keys_are_reported_once(self):
        model = self._model()
        with self.assertLogs("ov_backend", level="INFO") as captured:
            model.transcribe(np.zeros(WHISPER_SAMPLING_RATE, dtype=np.float32), condition_on_previous_text=True)
        self.assertIn("condition_on_previous_text", captured.output[0])
        # A second call with the same key logs nothing, so the log is not per-request noise.
        with self.assertNoLogs("ov_backend", level="INFO"):
            model.transcribe(np.zeros(WHISPER_SAMPLING_RATE, dtype=np.float32), condition_on_previous_text=True)


if __name__ == "__main__":
    unittest.main()
