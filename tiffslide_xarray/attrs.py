from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence
from collections import UserDict
import numpy as np
from numpy.typing import ArrayLike


def raise_if_cycle(x, seen=set()):
    if id(x) in seen:
        raise ValueError("Found a cycle.")

    seen = {id(x)} | seen

    if isinstance(x, Sequence):
        for i in x:
            raise_if_cycle(i, seen)

    if isinstance(x, Mapping):
        for k in x:
            raise_if_cycle(x[k], seen)


class XarrayAttrsProxy(UserDict):
    attrs: dict[str, Any]
    prefix: list[str]

    def __init__(
        self,
        attrs: dict[str, Any],
        prefix: list[str] | None = None,
    ):
        self.data = attrs
        self._prefix: list[str] = prefix or []

    def __str__(self):
        return dict(self).__str__()

    def _repr_pretty_(self, p, cycle):
        from IPython.lib.pretty import _dict_pprinter_factory  # type: ignore

        #begin, end = "{", "}"
        begin = f"XarrayAttrsProxy(\n\tdata="
        end = f"\n\tprefix={self._prefix}\n)"

        f = _dict_pprinter_factory(begin, end)
        prefix = self._prefixed_key("")

        DICT = dict((k, v) for k, v in self.data.items() if k.startswith(prefix))
        
        p.text("XarrayAttrsProxy")
        p.begin_group(2, "(\n")
        p.begin_group(4, "  attrs=")
        p.pretty(DICT)
        p.end_group(4, "")
        if self._prefix:
          p.breakable()
          p.begin_group(4, "  prefix=")
          p.pretty(self._prefix)
          p.end_group(4, "")
        p.end_group(2, ")")

        return p



        

    def __repr__(self):
        return self.data.__repr__()

    def __getitem__(self, key):
        if not self.is_valid_key(key):
            raise KeyError

        for k, v in self.items():
            if k == key:
                return v

        raise KeyError

    def __delitem__(self, key: str) -> None:
        assert self.is_valid_key(key)

        prefixed_key = self._prefixed_key(key)

        if prefixed_key in self.data:
            del self.data[prefixed_key]

        for k in list(self.data):
            if k.startswith(prefixed_key + "."):
                del self.data[k]

    def is_valid_key(self, key):
        if not isinstance(key, str):
            return False
        if "." in key:
            return False
        if "[" in key[1:]:
            return False
        return True

    def is_valid_value_container(self, value: tuple | list):
        return np.array(value).dtype != np.dtype("O")

    def _prefixed_key(self, key: str | list[str]) -> str:
        if type(key) == str:
            key = [key]

        assert type(key) == list
        new_key = ".".join(self._prefix + key)
        new_key = new_key.replace(".[", "[")
        return new_key

    def _split_key(self, key):
        key = key.replace("[", ".[")
        keys = key.split(".")
        return keys

    def __contains__(self, key):
        for k in self.keys():
            if k == key:
                return True
        return False

    def items(self):
        assert self.data
        out = {}
        for key, v in list(self.data.items()):
            ks = self._split_key(key)

            if tuple(self._prefix) != tuple(ks[: len(self._prefix)]):
                continue

            ks = ks[len(self._prefix) :]
            N = len(ks)

            if N == 1:  # return value
                out[ks[0]] = v

            if N == 2:  # return proxy to nested dictionary
                out[ks[0]] = XarrayAttrsProxy(self.data, self._prefix + ks[:1])

        return out.items()

    def values(self):
        yield from (v for _, v in self.items())

    def keys(self):
        yield from (k for k, _ in self.items())

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(list(self.items()))

    def __setitem__(self, key: str, value: str | float | int | ArrayLike | Mapping):
        assert self.is_valid_key(key)

        if isinstance(value, Mapping):
            if raise_if_cycle(value):
                raise ValueError("Cannot assign mapping with a cycle.")

            del self[key]

            subref = self.__class__(self.data, self._prefix + [key])

            subref.update(value)
            return

        elif isinstance(value, tuple):
            assert self.is_valid_value_container(value)
        elif isinstance(value, list):
            assert self.is_valid_value_container(value)

        prefixed_key = self._prefixed_key(key)
        self.data[prefixed_key] = value

    def update(self, other: Mapping[str, Any]):
        if raise_if_cycle(other):
            raise ValueError("Cannot update from mapping with a cycle.")        

        for k in other:
            self[k] = other[k]
