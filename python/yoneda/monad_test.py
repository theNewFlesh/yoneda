from functools import partial
import unittest

from lunchbox.enforce import EnforceError

import yoneda.monad as ym
from yoneda.monad import (
    iwrap, ifmap, iapp, ibind, iright, ifail, icurry, idot
)
# ------------------------------------------------------------------------------


class MonadFunctionTests(unittest.TestCase):
    def test_enforce_monad(self):
        ym.enforce_monad(ym.Monad)
        ym.enforce_monad(ym.Monad(42))

        class TestMonad(ym.Monad):
            pass

        ym.enforce_monad(TestMonad)
        ym.enforce_monad(TestMonad(42))

        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.enforce_monad('foo')

    def test_wrap(self):
        result = ym.wrap(ym.Monad, 9)
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(result._data, 9)

    def test_wrap_errors(self):
        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.wrap('foo', 9)

    def test_unwrap(self):
        monad = ym.wrap(ym.Monad, 9)
        result = ym.unwrap(monad)
        self.assertEqual(result, 9)

    def test_unwrap_errors(self):
        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.unwrap('foo')

    def test_fmap(self):
        monad = ym.wrap(ym.Monad, 2)
        func = lambda x: x + 2
        result = ym.fmap(func, monad)
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(ym.unwrap(result), 4)

    def test_fmap_errors(self):
        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.fmap(lambda x: x, 'foo')

    def test_app(self):
        monad = ym.wrap(ym.Monad, 2)
        func = ym.Monad(lambda x: x + 2)
        result = ym.app(func, monad)
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(ym.unwrap(result), 4)

    def test_app_errors(self):
        expected = 'bar is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.app('bar', ym.Monad(2))

        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.app(ym.Monad(lambda x: x + 2), 'foo')

    def test_bind(self):
        monad = ym.wrap(ym.Monad, 2)
        func = lambda x: ym.Monad(x + 2)
        result = ym.bind(func, monad)
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(ym.unwrap(result), 4)

    def test_bind_errors(self):
        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.bind(lambda x: ym.Monad(x + 2), 'foo')

    def test_right(self):
        a = ym.wrap(ym.Monad, 2)
        expected = ym.wrap(ym.Monad, 4)
        result = ym.right(a, expected)
        self.assertIs(result, expected)

    def test_right_errors(self):
        monad = ym.Monad(10)

        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.right('foo', monad)

        with self.assertRaisesRegex(EnforceError, expected):
            ym.right(monad, 'foo')

    def test_fail(self):
        error = SyntaxError('foo')
        result = ym.fail(ym.Monad, error)
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(ym.unwrap(result), error)

    def test_fail_errors(self):
        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.fail('foo', SyntaxError('bar'))

        expected = 'Error must be an instance of Exception. Given value: bar'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.fail(ym.Monad('foo'), 'bar')


class MonadInfixFunctionTests(unittest.TestCase):
    def test_wrap_infix(self):
        monad = ym.Monad
        data = 99
        result = monad |iwrap| data  # noqa: E225
        expected = ym.wrap(monad, data)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(ym.unwrap(result), ym.unwrap(expected))

    def test_fmap_infix(self):
        func = lambda x: x + 7
        monad = ym.Monad(14)
        result = func |ifmap| monad  # noqa: E225
        expected = ym.fmap(func, monad)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(ym.unwrap(result), ym.unwrap(expected))

    def test_app_infix(self):
        monad_func = ym.Monad(lambda x: x * 2)
        monad = ym.Monad(5)
        result = monad_func |iapp| monad  # noqa: E225
        expected = ym.app(monad_func, monad)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(ym.unwrap(result), ym.unwrap(expected))

    def test_bind_infix(self):
        class Foo(ym.Monad):
            pass

        func = Foo.wrap
        monad = ym.Monad(99)
        result = func |ibind| monad  # noqa: E225
        expected = ym.bind(func, monad)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(ym.unwrap(result), ym.unwrap(expected))

    def test_right_infix(self):
        monad_a = ym.Monad('a')
        monad_b = ym.Monad('b')
        result = monad_a |iright| monad_b  # noqa: E225
        expected = ym.right(monad_a, monad_b)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(ym.unwrap(result), ym.unwrap(expected))

    def test_fail_infix(self):
        monad = ym.Monad
        error = SyntaxError('foobar')
        result = monad |ifail| error  # noqa: E225
        expected = ym.fail(monad, error)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(ym.unwrap(result), ym.unwrap(expected))

    def test_curry(self):
        func = lambda x, y: x + y
        cur = func |icurry| 'a'  # noqa: E225
        self.assertIsInstance(cur, partial)
        self.assertEqual(cur('b'), 'ab')

        result = func |icurry| 1 |icurry| 2  # noqa: E225
        self.assertEqual(result(), 3)

    def test_dot(self):
        fa = lambda x: dict(a='b')[x]
        fb = lambda x: dict(b='c')[x]

        self.assertIsInstance(ym.dot(fb, fa), partial)
        self.assertIsInstance(fb |idot| fa, partial)  # noqa: E225

        self.assertEqual(ym.dot(fb, fa)('a'), 'c')
        self.assertEqual((fb |idot| fa)('a'), 'c')  # noqa: E225

        with self.assertRaises(KeyError):
            ym.dot(fb, fa)('b')

        with self.assertRaises(KeyError):
            f = fb |idot| fa  # noqa: E225
            f('b')

        fx = lambda x: x + 'a'
        fy = lambda x: x + 'b'

        self.assertEqual(ym.dot(fy, fx)('x'), 'xab')
        self.assertEqual((fy |idot| fx)('x'), 'xab')  # noqa: E225

        fa = lambda x: dict(a='b')[x]
        fb = lambda x: dict(b='c')[x]
        fc = lambda x: dict(c='d')[x]

        f = fc |idot| fb |idot| fa  # noqa: E225
        self.assertEqual(f('a'), 'd')

    def test_partial_dot(self):
        result = ym.partial_dot(lambda x: x + 2)
        self.assertIsInstance(result, partial)

        result = ym.partial_dot(lambda x: x + 2)(lambda x: x + 3)(5)
        self.assertEqual(result, 10)

        result = ym.partial_dot(
            lambda x: f'1st-{x}')(lambda x: f'2nd-{x}')('3rd')
        self.assertEqual(result, '1st-2nd-3rd')

    def test_catch(self):
        result = ym.catch(ym.Monad, lambda x: x + 2)(1)
        self.assertEqual(result, 3)

        # error
        result = ym.catch(ym.Monad, lambda x: x + 'bar')(1)
        self.assertIsInstance(result, ym.Monad)
        self.assertIsInstance(result.unwrap(), TypeError)

    def test_catch_monadic_funcs(self):
        # fmap
        func = lambda x: x + 'bar'
        result = ym.catch(ym.Monad, ym.fmap)(func, ym.Monad('foo'))
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(result.unwrap(), 'foobar')

        # fmap error
        result = ym.catch(ym.Monad, ym.fmap)(func, ym.Monad(1))
        self.assertIsInstance(result, ym.Monad)
        self.assertIsInstance(result.unwrap(), TypeError)

        # bind
        func = lambda x: ym.Monad(x + 'bar')
        result = ym.catch(ym.Monad, ym.bind)(func, ym.Monad('foo'))
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(result.unwrap(), 'foobar')

        # bind error
        result = ym.catch(ym.Monad, ym.bind)(func, ym.Monad(1))
        self.assertIsInstance(result, ym.Monad)
        self.assertIsInstance(result.unwrap(), TypeError)

        # app
        func = ym.Monad(lambda x: x + 'bar')
        result = ym.catch(ym.Monad, ym.app)(func, ym.Monad('foo'))
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(result.unwrap(), 'foobar')

        # app error
        result = ym.catch(ym.Monad, ym.app)(func, ym.Monad(1))
        self.assertIsInstance(result, ym.Monad)
        self.assertIsInstance(result.unwrap(), TypeError)

    def test_catch_errors(self):
        expected = 'foo is not a subclass or instance of Monad.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.catch('foo', 99)


class MonadTests(unittest.TestCase):
    def test_init(self):
        result = ym.Monad(42)
        self.assertEqual(result._data, 42)

    def test_wrap(self):
        result = ym.Monad.wrap(42)
        self.assertIsInstance(result, ym.Monad)

    def test_unwrap(self):
        result = ym.Monad.wrap(42).unwrap()
        self.assertEqual(result, 42)

    def test_fmap(self):
        result = ym.Monad.wrap(42).fmap(lambda x: x + 10)
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(result.unwrap(), 52)

    def test_app(self):
        result = ym.Monad.wrap(42).app(ym.Monad.wrap(lambda x: x + 10))
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(result.unwrap(), 52)

    def test_bind(self):
        result = ym.Monad.wrap(42).bind(lambda x: ym.Monad.wrap(x + 10))
        self.assertIsInstance(result, ym.Monad)
        self.assertEqual(result.unwrap(), 52)

    def test_right(self):
        expected = ym.Monad.wrap(99)
        result = ym.Monad \
            .wrap(42) \
            .right(expected)
        self.assertIs(result, expected)
        self.assertIs(result.unwrap(), expected.unwrap())

    def test_fail(self):
        error = SyntaxError('foo')
        result = ym.Monad.wrap(42).fail(error)
        self.assertIsInstance(result, ym.Monad)
        self.assertIs(result.unwrap(), error)

    def test_succeed(self):
        ym.succeed(ym.Monad, 42)

        expected = 'Error must not be an instance of Exception. '
        expected += 'Given value: foo.'
        with self.assertRaisesRegex(EnforceError, expected):
            ym.succeed(ym.Monad, SyntaxError('foo'))

    def test_repr(self):
        class Foo(ym.Monad):
            pass

        result = str(Foo(99))
        self.assertEqual(result, 'Foo(99)')

    def test_and(self):
        m = ym.Monad(99)
        func = lambda x: x + 2
        result = m & func
        expected = m.fmap(func)
        self.assertIsInstance(result, ym.Monad)
        self.assertIs(result.unwrap(), expected.unwrap())

    def test_xor(self):
        m = ym.Monad.wrap(99)
        func = ym.Monad(lambda x: x + 10)
        result = m ^ func
        expected = m.app(func)
        self.assertIsInstance(result, ym.Monad)
        self.assertIs(result.unwrap(), expected.unwrap())

    def test_rshift(self):
        class Foo(ym.Monad):
            pass

        m = ym.Monad(99)
        result = m >> Foo
        expected = m.bind(Foo)
        self.assertIsInstance(result, Foo)
        self.assertIs(result.unwrap(), expected.unwrap())

    # LAWS----------------------------------------------------------------------
    def test_bind_left_identity(self):
        # Haskell: return a >>= h     =  ha
        # Python:  wrap(a).bind(func) == func(a)
        class TestMonad(ym.Monad):
            pass

        x = 1
        func = TestMonad.wrap
        result = ym.Monad.wrap(x).bind(func)
        expected = func(x)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result._data, expected._data)

    def test_bind_right_identity(self):
        # Haskell: m >>= return   =  m
        # Python:  m.bind(m.wrap) == m

        m = ym.Monad.wrap(99)
        result = m.bind(m.wrap)
        expected = m
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result._data, expected._data)

    def test_bind_associativity(self):
        # Haskell: (m >>= g) >>= h = m >>= (\x -> g x >>= h)
        # Python: m.bind(g).bind(h) == m.bind(lambda x: g(x).bind(h))

        class TestMonad1(ym.Monad):
            pass

        class TestMonad2(ym.Monad):
            pass

        m = ym.Monad(99)
        g = TestMonad1
        h = TestMonad2
        result = m.bind(g).bind(h)
        expected = m.bind(lambda x: g(x).bind(h))
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result._data, expected._data)

    def test_fmap_identity(self):
        # Haskell: fmap id = id
        # Python:  m.fmap(lambda x: x) == lambda x: x

        identity = lambda x: x
        m = ym.Monad.wrap(99)
        result = m.fmap(identity)
        expected = identity(m)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result._data, expected._data)

    def test_fmap_distributivity(self):
        # Haskell: fmap (g . h) = (fmap g) . (fmap h)
        # Python:  m.fmap(lambda x: h(g(x))) == m.fmap(g).fmap(h)

        m = ym.Monad.wrap(99)
        g = lambda x: x - 1
        h = lambda x: x * 2
        result = m.fmap(lambda x: h(g(x)))
        expected = m.fmap(g).fmap(h)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result._data, expected._data)

    def test_app_identity(self):
        # Haskell: pure id <*> v = v
        # Python:  m.wrap(v).app( m.wrap(lambda x: x) )

        m = ym.Monad.wrap(99)
        result = m.app(m.wrap(lambda x: x))
        self.assertEqual(result.__class__, m.__class__)
        self.assertEqual(result._data, m._data)

    def test_app_homomorphism(self):
        # Haskell: (pure f) <*> (pure x) = pure (f x)
        # Python:  m.wrap(x).app(m.wrap(func)) == m.wrap(func(x))

        m = ym.Monad
        func = lambda x: x + 2
        x = 2
        result = m.wrap(x).app(m.wrap(func))
        expected = m.wrap(func(x))
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result._data, expected._data)

    def test_app_interchange(self):
        # Haskell: u <*> (pure y) = pure (\f -> f y) <*> u
        # Python:  m(y).app(m.wrap(u)) == m(u).app(m.wrap(lambda f: f(y)))
        #          u = lambda x: 42

        m = ym.Monad
        u = lambda x: 42
        y = 5
        result = m.wrap(y).app(m.wrap(u))
        expected = m.wrap(u).app(m.wrap(lambda f: f(y)))
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result._data, expected._data)

    def test_app_composition_left(self):
        # Haskell: u <*> (v <*> w) = pure (.) <*> u <*> v <*> w
        # left expression:: u <*> (v <*> w)

        M = ym.Monad
        u = M(lambda x: x - 1)
        v = M(lambda x: x - 2)
        w = M(3)

        result = u |iapp| (v |iapp| w)  # noqa: E225
        exp = M(0)
        self.assertEqual(result.__class__, exp.__class__)
        self.assertEqual(result.unwrap(), exp.unwrap())
        exp = w.app(v).app(u)
        self.assertEqual(result.unwrap(), exp.unwrap())

    def test_app_composition_right(self):
        # Haskell: u <*> (v <*> w) = pure (.) <*> u <*> v <*> w
        # right expression :: pure (.) <*> u <*> v <*> w
        #                . :: (b -> c) -> (a -> b) -> (a -> c)
        #             pure :: a -> f a
        #              app :: m(a -> b)-> ma -> mb

        u = lambda x: x - 1
        v = lambda x: x - 2
        w = 3
        assert ym.dot(u, v)(w) == u(v(w))

        app = ym.app
        d = lambda x: partial(ym.dot, x)
        M = ym.Monad
        u = M(lambda x: x - 1)
        v = M(lambda x: x - 2)
        w = M(3)

        p = M.wrap(d)
        x0 = app(p, u)
        x1 = app(x0, v)
        result = app(x1, w)
        expected = M(0)
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result.unwrap(), expected.unwrap())

        result = app(app(app(M.wrap(d), u), v), w)
        self.assertEqual(result.unwrap(), expected.unwrap())

        result = w.app(v.app(u.app(M.wrap(d))))
        self.assertEqual(result.unwrap(), expected.unwrap())

        result = w ^ (v ^ (u ^ M.wrap(d)))
        self.assertEqual(result.unwrap(), expected.unwrap())

        result = M.wrap(d) |iapp| u |iapp| v |iapp| w  # noqa: E225
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result.unwrap(), expected.unwrap())

    def test_app_composition(self):
        # Haskell: u <*> (v <*> w) = pure (.) <*> u <*> v <*> w

        M = ym.Monad
        d = ym.partial_dot
        u = M(lambda x: x - 1)
        v = M(lambda x: x - 2)
        w = M(3)

        result = u |iapp| (v |iapp| w)  # noqa: E225
        expected = M(d) |iapp| u |iapp| v |iapp| w  # noqa: E225
        self.assertEqual(result.__class__, expected.__class__)
        self.assertEqual(result.unwrap(), expected.unwrap())
