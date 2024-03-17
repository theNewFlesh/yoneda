from typing import Callable, Generic, TypeVar  # noqa: F401

import math

import yoneda.monad as sgm
from yoneda.monad import Monad

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
# ------------------------------------------------------------------------------


class Maybe(Monad, Generic[A]):
    @classmethod
    def just(cls, value):
        # type: (A) -> Maybe[A]
        '''
        Just constructor for Maybe class.

        Args:
            value (object): Non-null value.

        Returns:
            Maybe: Maybe monad of value.
        '''
        return cls(value)

    @classmethod
    def nothing(cls):
        # type: () -> Maybe
        '''
        Nothing constructor for Maybe class.

        Returns:
            Maybe: Nothing monad.
        '''
        return cls(None)

    def __repr__(self):
        # type: () -> str
        '''String representation of monad.'''
        if self.state == 'just':
            return f'Just({self._data})'
        return 'Nothing'

    @property
    def state(self):
        # type: () -> str
        '''State of monad. Either just or nothing.'''
        data = self._data
        if data is None or math.isnan(data):
            return 'nothing'
        return 'just'
# ------------------------------------------------------------------------------


class Try(Monad, Generic[A]):
    @classmethod
    def success(cls, value):
        # type: (A) -> Try[A]
        '''
        Success constructor for Try class.

        Args:
            value (object): Non-error value.

        Returns:
            Maybe: Try monad of value.
        '''
        return sgm.succeed(cls, value)

    @classmethod
    def failure(cls, error):
        # type: (Exception) -> Try[Exception]
        '''
        Success constructor for Try class.

        Args:
            error (Exception): Error.

        Returns:
            Maybe: Try monad of error.
        '''
        return sgm.fail(cls, error)

    def __repr__(self):
        # type: () -> str
        '''String representation of monad.'''
        return f'{self.state.capitalize()}({self._data})'

    @property
    def state(self):
        # type: () -> str
        '''State of monad. Either success or failure.'''
        if isinstance(self._data, Exception):
            return 'failure'
        return 'success'

    def fmap(self, func):
        # type: (Callable[[A], B]) -> Try[B | Exception]
        '''
        Functor map: (A -> B) -> MB

        Given a function A to B, return a Monad of B (MB).
        Example: m.fmap(lambda x: x + 2)

        Args:
            func (function): Function (A -> B).

        Returns:
            Try[B]: Try Monad of B.
        '''
        try:
            return super().fmap(func)  # type: ignore
        except Exception as error:
            return self.fail(error)

    def bind(self, func):
        # type: (Callable[[A], Monad[B]]) -> Try[B | Exception]
        '''
        Bind: (A -> MB) -> MB

        Given a function A to MB, return a Monad of B (MB).

        Args:
            func (function): Function (A -> MB).

        Returns:
            Try[B]: Try Monad of B.
        '''
        try:
            return super().bind(func)  # type: ignore
        except Exception as error:
            return self.fail(error)

    def app(self, monad_func):
        # type: (Monad[Callable[[A], B]]) -> Try[B | Exception]
        '''
        Applicative: M(A -> B) -> MB

        Given a Monad of a function A to B, return a Monad of B (MB).

        Args:
            monad_func (Monad): Monad of function (A -> B).

        Returns:
            Try[B]: Try Monad of B.
        '''
        try:
            return super().app(monad_func)  # type: ignore
        except Exception as error:
            return self.fail(error)
