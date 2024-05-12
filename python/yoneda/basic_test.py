import math
import unittest

from lunchbox.enforce import EnforceError

import yoneda.basic as yb
import yoneda.monad as ym
# ------------------------------------------------------------------------------


class MaybeTests(unittest.TestCase):
    def test_just(self):
        result = yb.Maybe.just(5)
        self.assertIsInstance(result, yb.Maybe)
        self.assertEqual(result.unwrap(), 5)

    def test_nothing(self):
        result = yb.Maybe.nothing()
        self.assertIsInstance(result, yb.Maybe)

    def test_repr(self):
        result = str(yb.Maybe.just(5))
        self.assertEqual(result, 'Just(5)')

        result = str(yb.Maybe.nothing())
        self.assertEqual(result, 'Nothing')

    def test_state(self):
        result = yb.Maybe.just(5).state
        self.assertEqual(result, 'just')

        result = yb.Maybe.nothing().state
        self.assertEqual(result, 'nothing')

        result = yb.Maybe(None).state
        self.assertEqual(result, 'nothing')

        result = yb.Maybe(math.nan).state
        self.assertEqual(result, 'nothing')


class TryTests(unittest.TestCase):
    def test_success(self):
        result = yb.Try.success(5)
        self.assertIsInstance(result, yb.Try)

        with self.assertRaises(EnforceError):
            yb.Try.success(ValueError('foo'))

    def test_failure(self):
        result = yb.Try.failure(ValueError('foo'))
        self.assertIsInstance(result, yb.Try)

        with self.assertRaises(EnforceError):
            yb.Try.failure(99)

    def test_repr(self):
        result = str(yb.Try.success(5))
        self.assertEqual(result, 'Success(5)')

        result = str(yb.Try.failure(ValueError('foo')))
        expected = 'Failure(' + str(ValueError('foo')) + ')'
        self.assertEqual(result, expected)

    def test_state(self):
        result = yb.Try(5).state
        self.assertEqual(result, 'success')

        result = yb.Try(ValueError('foo')).state
        self.assertEqual(result, 'failure')

    def test_fmap(self):
        result = yb.Try(2).fmap(lambda x: x + 1)
        self.assertEqual(result.unwrap(), 3)

        # error
        result = yb.Try(2).fmap(lambda x: x + 'foo')
        self.assertEqual(type(result.unwrap()), TypeError)

    def test_bind(self):
        result = yb.Try(2).bind(lambda x: ym.Monad(x + 1))
        self.assertEqual(result.unwrap(), 3)
        self.assertEqual(result.__class__.__name__, 'Monad')

        # error
        result = yb.Try(2).bind(lambda x: ym.Monad(x + 'foo'))
        self.assertEqual(type(result.unwrap()), TypeError)

    def test_app(self):
        result = yb.Try(2).app(ym.Monad(lambda x: x + 1))
        self.assertEqual(result.unwrap(), 3)

        # error
        result = yb.Try(2).app(ym.Monad(lambda x: x + 'foo'))
        self.assertEqual(type(result.unwrap()), TypeError)
