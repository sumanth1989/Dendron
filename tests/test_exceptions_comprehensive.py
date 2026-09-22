"""
Comprehensive unit tests for dendron.exceptions.
Verifies the inheritance hierarchy and backwards-compatibility of custom exceptions.
"""

import unittest
from dendron.exceptions import (
    DendronError,
    NodeNotFoundError,
    InvalidTransitionError,
    ToolValidationError,
    SecurityClearanceError,
    CycleDetectedError,
    ToolExecutionError,
)


class TestExceptionsComprehensive(unittest.TestCase):

    def test_node_not_found_error_inheritance(self):
        err = NodeNotFoundError("Node 'x' not found")
        self.assertIsInstance(err, DendronError)
        self.assertIsInstance(err, KeyError)
        self.assertIn("Node 'x' not found", str(err))

    def test_validation_and_transition_errors(self):
        v_err = ToolValidationError("Placeholder detected")
        self.assertIsInstance(v_err, DendronError)
        self.assertIsInstance(v_err, ValueError)

        t_err = InvalidTransitionError("Cannot transition")
        self.assertIsInstance(t_err, DendronError)
        self.assertIsInstance(t_err, ValueError)

        c_err = CycleDetectedError("Cycle detected")
        self.assertIsInstance(c_err, DendronError)
        self.assertIsInstance(c_err, ValueError)

    def test_security_and_execution_errors(self):
        s_err = SecurityClearanceError("Confirmation required")
        self.assertIsInstance(s_err, DendronError)
        self.assertIsInstance(s_err, PermissionError)

        e_err = ToolExecutionError("Handler crashed")
        self.assertIsInstance(e_err, DendronError)
        self.assertIsInstance(e_err, RuntimeError)


if __name__ == "__main__":
    unittest.main()
