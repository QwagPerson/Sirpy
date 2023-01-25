import unittest

import numpy as np

from sirpy.models.abstractModel import EmptyModel, State, Transition


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        # Lets create an empty model to use for testing
        self.m = EmptyModel(
            "test",
            hyper_params={
                "a": 1,
                "b": np.array([1, 2, 3]),
            },
            train_params={
                "c": 1,
                "d": np.ones((3)),
            },
            static_params={
                "e": 1,
                "f": np.ones((2, 1, 3)),
            }
        )

    def test_state_and_transitions(self):
        # Checking everything is empty at first
        self.assertEqual(0, len(self.m.states))
        self.assertEqual(0, len(self.m.transitions))

        # Adding a state
        self.m.add_state(State("a"))
        self.assertEqual(1, len(self.m.states))
        self.assertEqual(0, len(self.m.transitions))

        # Adding a transition
        self.m.add_transition(Transition("a->b", "a", "b", lambda t, y, p: 0))
        self.assertEqual(1, len(self.m.states))
        self.assertEqual(1, len(self.m.transitions))

        # Adding 3 states with add_states
        self.m.add_states([State("b"), State("c"), State("d")])
        self.assertEqual(4, len(self.m.states))

        # Adding 3 transitions with add_transitions
        self.m.add_transitions([Transition("a->b", "a", "b", lambda t, y, p: 0),
                                Transition("b->c", "b", "c", lambda t, y, p: 0),
                                Transition("c->d", "c", "d", lambda t, y, p: 0)])
        self.assertEqual(4, len(self.m.states))

    def test_save_shapes_and_sizes_of_train_params(self):
        self.m.save_shapes_and_sizes_of_train_params()
        self.assertEqual([1, (3,)], self.m.shapes)
        self.assertEqual([1, 3], self.m.sizes)

    def test_get_train_params_as_list(self):
        self.m.save_shapes_and_sizes_of_train_params()
        self.assertEqual([1] + [1.0] * 3, self.m.get_train_params_as_list())

    def test_set_train_params_from_list(self):
        self.m.save_shapes_and_sizes_of_train_params()
        self.m.set_train_params_from_list([0, 10, 20, 30])
        self.assertEqual(0, self.m.train_params["c"])
        self.assertTrue(np.array_equal(np.array([10, 20, 30]), self.m.train_params["d"]))

    def get_param_tests(self):
        self.assertEqual(1, self.m.get_param("a"))
        self.assertTrue(np.array_equal(np.array([1, 2, 3]), self.m.get_param("b")))
        self.assertEqual(1, self.m.get_param("c"))
        self.assertTrue(np.array_equal(np.ones((1, 2, 3)), self.m.get_param("d")))
        self.assertEqual(1, self.m.get_param("e"))
        self.assertTrue(np.array_equal(np.ones((2, 1, 3)), self.m.get_param("f")))

if __name__ == '__main__':
    unittest.main()
