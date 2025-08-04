import unittest
import numpy as np
from motor_n_dig import MotorN

class TestMotorN(unittest.TestCase):
    def setUp(self):
        self.motor = MotorN(dim=10)

    def test_init(self):
        self.assertEqual(self.motor.rho.shape, (10, 10))
        self.assertTrue(np.all(self.motor.rho >= 0))
        self.assertTrue(np.all(self.motor.rho <= 0.1))

    def test_evolucionar(self):
        rho_before = self.motor.rho.copy()
        self.motor.evolucionar()
        self.assertEqual(self.motor.rho.shape, (10, 10))
        self.assertFalse(np.allclose(rho_before, self.motor.rho))

    def test_inyectar(self):
        x, y = 5, 5
        valor_antes = self.motor.rho[y, x]
        self.motor.inyectar(x, y, intensidad=1.0)
        self.assertGreaterEqual(self.motor.rho[y, x], valor_antes)
        self.assertLessEqual(self.motor.rho[y, x], 1.0)

    def test_reiniciar(self):
        self.motor.inyectar(0, 0, intensidad=1.0)
        self.motor.reiniciar()
        self.assertTrue(np.all(self.motor.rho <= 0.1))
        self.assertTrue(np.all(self.motor.rho >= 0))

    def test_obtener_rho(self):
        rho = self.motor.obtener_rho()
        self.assertTrue(np.allclose(rho, self.motor.rho))

    def test_obtener_entropia_global(self):
        entropia = self.motor.obtener_entropia_global()
        self.assertIsInstance(entropia, float)
        self.assertGreaterEqual(entropia, 0)

if __name__ == "__main__":
    unittest.main()
