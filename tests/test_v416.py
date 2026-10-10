import unittest
from v416 import AnonymousEventWorld, SelfOrganizingLearner, reconstruct_steps, benchmark

class TestV416(unittest.TestCase):
    def test_raw_stream_roundtrip(self):
        w=AnonymousEventWorld(1)
        stream=w.raw_rollout(120,2)
        steps=reconstruct_steps(stream,w.action_tokens)
        self.assertGreaterEqual(len(steps),115)
        self.assertTrue(all(isinstance(t,float) for t,_ in stream))

    def test_learner_uses_anonymous_stream(self):
        w=AnonymousEventWorld(4)
        stream=w.raw_rollout(500,5)
        learner=SelfOrganizingLearner(w.action_tokens).fit(stream)
        self.assertGreater(len(learner.mapping),20)
        self.assertLess(len(learner.discovered_groups()),len(learner.mapping))

    def test_small_gate_screen(self):
        s,_=benchmark(4)
        self.assertGreater(s["prediction_gain_over_raw"],0.08)
        self.assertGreater(s["action_ablation_gap"],0.08)
        self.assertGreater(s["shuffle_gap"],0.08)
        self.assertGreater(s["control_abs"],0.70)

if __name__ == '__main__': unittest.main()
