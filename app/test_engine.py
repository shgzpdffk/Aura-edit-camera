"""Behavioral checks: no camera is accessed. Run python -m unittest -v."""
import unittest
import cv2
import numpy as np

from engine import MotionDetector, Observation, ReplayBuffer, SelfEdit, TriggerGate


class MotionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.frame = rng.integers(40, 180, (360, 640, 3), dtype=np.uint8)

    def test_static_face_and_global_exposure_do_not_trigger(self):
        d = MotionDetector()
        d.update(self.frame, 1.0, (100, 45, 70, 70))
        self.assertEqual(d.update(self.frame, 1.2, (100, 45, 70, 70)).motion, 0)
        brighter = (self.frame.astype(np.int16) + 15).astype(np.uint8)
        self.assertLess(d.update(brighter, 1.4, (100, 45, 70, 70)).motion, 1)

    def test_expression_and_head_motion_are_seen(self):
        d = MotionDetector()
        d.update(self.frame, 1.0, (100, 45, 70, 70))
        expression = self.frame.copy()
        expression[130:160, 220:300] = 240
        self.assertGreater(d.update(expression, 1.2, (100, 45, 70, 70)).motion, 4)
        moved = np.roll(expression, 24, axis=1)
        self.assertGreater(d.update(moved, 1.4, (112, 45, 70, 70)).motion, 4)

    def test_moving_empty_scene_does_not_trigger(self):
        d = MotionDetector()
        d.update(self.frame, 1.0, None)
        moved = np.roll(self.frame, 40, axis=0)
        self.assertEqual(d.update(moved, 1.2, None).motion, 0)

    def test_gate_debounce_cooldown_and_busy(self):
        gate = TriggerGate()
        action = Observation((1, 1, 50, 50), 45, True)
        quiet = Observation((1, 1, 50, 50), 0, True)
        self.assertFalse(gate.check(quiet, 1, buffered_seconds=3))
        self.assertFalse(gate.check(action, 2, buffered_seconds=3))
        self.assertTrue(gate.check(action, 2.1, buffered_seconds=3))
        self.assertFalse(gate.check(action, 3, buffered_seconds=3))
        self.assertFalse(gate.check(action, 7, busy=True, buffered_seconds=3))
        self.assertFalse(gate.check(action, 8, buffered_seconds=3))
        self.assertTrue(gate.check(action, 8.1, buffered_seconds=3))

    def test_buffer_bounded_and_timestamps_drive_edit(self):
        buffer = ReplayBuffer(seconds=6, fps=15)
        for i in range(360):
            buffer.push(np.roll(self.frame, i, axis=1), i / 30)
        self.assertLessEqual(len(buffer.frames), 92)
        self.assertLessEqual(buffer.duration, 6)
        edit = SelfEdit(buffer.snapshot(), variant=2)
        self.assertFalse(np.array_equal(edit.frame_at(0), edit.frame_at(2.1)))
        for t in (0, 1.3, 3.4, 4.6, 5.9):
            frame = edit.frame_at(t)
            self.assertEqual(frame.shape, (360, 640, 3))
            self.assertEqual(frame.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
