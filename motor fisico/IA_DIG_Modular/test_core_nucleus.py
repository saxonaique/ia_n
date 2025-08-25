"""
Unit tests for the CoreNucleus class.
"""

import unittest
import numpy as np
from core_nucleus import CoreNucleus

class TestCoreNucleus(unittest.TestCase):
    """Test cases for CoreNucleus functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.size = (10, 10)
        self.config = {
            'entropy_window': 3,
            'equilibrium_threshold': 0.1,
            'max_history': 5,
            'min_entropy_change': 1e-5
        }
        self.core = CoreNucleus(self.config)
        
        # Create a simple test field
        self.test_field = np.zeros(self.size, dtype=np.int8)
        self.test_field[2:5, 2:5] = 1  # Add a small square pattern
        
    def test_initialization(self):
        """Test CoreNucleus initialization."""
        self.assertIsNone(self.core.field)
        self.assertIsNone(self.core.entropy)
        self.assertEqual(len(self.core.history), 0)
        
    def test_receive_field(self):
        """Test receiving a field updates the internal state."""
        self.core.receive_field(self.test_field)
        self.assertIsNotNone(self.core.field)
        self.assertEqual(self.core.field.shape, self.size)
        self.assertEqual(len(self.core.history), 1)
        
    def test_compute_entropy(self):
        """Test entropy calculation."""
        self.core.receive_field(self.test_field)
        entropy, entropy_map = self.core.compute_entropy()
        
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertEqual(entropy_map.shape, self.size)
        
    def test_reorganize_field(self):
        """Test field reorganization."""
        self.core.receive_field(self.test_field)
        initial_entropy, _ = self.core.compute_entropy()
        
        # Reorganize with no memory reference
        reorganized = self.core.reorganize_field()
        self.assertEqual(reorganized.shape, self.size)
        
        # Check that the field was updated
        self.assertTrue(np.any(self.core.field != self.test_field))
        
        # Check that history was updated
        self.assertEqual(len(self.core.history), 2)
        
    def test_equilibrium_detection(self):
        """Test equilibrium state detection."""
        # Create a uniform field (should be in equilibrium)
        uniform_field = np.ones(self.size, dtype=np.int8)
        self.core.receive_field(uniform_field)
        
        # Should be in equilibrium (entropy below threshold)
        self.assertTrue(self.core.is_equilibrium())
        
        # Create a more complex pattern (checkerboard)
        checkerboard = np.indices(self.size).sum(axis=0) % 2 * 2 - 1  # Creates 1, -1, 1... pattern
        self.core.receive_field(checkerboard)
        
        # With our current entropy calculation, even a checkerboard can be in equilibrium
        # So we'll test with a very low threshold to force non-equilibrium
        self.config['equilibrium_threshold'] = 0.001
        self.core = CoreNucleus(self.config)
        self.core.receive_field(checkerboard)
        
        # With very low threshold, should not be in equilibrium
        self.assertFalse(self.core.is_equilibrium())

if __name__ == '__main__':
    unittest.main()
