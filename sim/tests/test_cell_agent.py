import unittest

from digital_cell_sim.cell import DigitalCellAgent


class DigitalCellAgentTests(unittest.TestCase):
    def test_online_predictions_use_only_previous_observations(self):
        agent = DigitalCellAgent(window=5)
        first = agent.observe(1.0, 0.5)
        second = agent.observe(3.0, 0.7)
        third = agent.observe(6.0, 0.9)
        self.assertIsNone(first.predicted_time)
        self.assertIsNone(first.predicted_length)
        self.assertIsNone(second.predicted_time)
        self.assertAlmostEqual(second.predicted_length, 0.5)
        self.assertAlmostEqual(third.predicted_time, 5.0)
        self.assertAlmostEqual(third.predicted_length, 0.6)

    def test_memory_is_bounded(self):
        agent = DigitalCellAgent(memory_capacity=3)
        for i in range(8):
            agent.observe(float(i), 0.5 + i * 0.01)
        self.assertEqual(len(agent.memory), 3)
        self.assertEqual(len(agent.experiences), 8)
        self.assertEqual(agent.memory[0].time, 5.0)

    def test_hidden_state_is_not_in_cell_memory_contract(self):
        agent = DigitalCellAgent()
        agent.observe(1.0, 0.7)
        exported = agent.export()
        self.assertEqual(exported["observation_contract"], ["time", "length"])
        self.assertNotIn("rod_id", str(exported))
        self.assertNotIn("position", str(exported))
        self.assertNotIn("velocity", str(exported))

    def test_invalid_and_out_of_order_observations_fail(self):
        agent = DigitalCellAgent()
        with self.assertRaises(ValueError):
            agent.observe(1.0, 0.0)
        agent.observe(2.0, 0.5)
        with self.assertRaises(ValueError):
            agent.observe(1.0, 0.6)


if __name__ == "__main__":
    unittest.main()
