import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er_partial_supervision import exterior_background, masked_bce_dice


class BackgroundTest(unittest.TestCase):
    def test_enclosed_dark_hole_is_not_exterior(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('SciPy unavailable')
        image = np.ones((128,128))*100
        image[:,:24] = 0
        image[48:80,48:80] = 0
        positive = np.zeros(image.shape,bool)
        positive[20:25,20:25] = True
        bg, _ = exterior_background(image,image,positive)
        self.assertTrue(bg.any())
        self.assertFalse(bg[55:70,55:70].any())
        self.assertFalse((bg & positive).any())


class LossTest(unittest.TestCase):
    def test_ignore_pixels_have_zero_gradient(self):
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest('TensorFlow unavailable')
        labels = tf.constant([0.,1.,255.])
        p = tf.Variable([.2,.8,.3])
        with tf.GradientTape() as tape:
            loss = masked_bce_dice(labels,p)
        self.assertEqual(float(tape.gradient(loss,p)[2]),0.)
        self.assertAlmostEqual(float(loss),float(masked_bce_dice(labels,tf.constant([.2,.8,.99]))),places=6)
        self.assertEqual(float(masked_bce_dice(tf.constant([255.]),tf.constant([.5]))),0.)
        self.assertLess(float(masked_bce_dice(labels,tf.constant([.01,.99,.5]))),float(loss))


if __name__ == '__main__':
    unittest.main()
