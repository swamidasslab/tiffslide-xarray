from __future__ import annotations

from typing import NamedTuple, Any, Callable, Sequence, Union
import numpy as np

import tree

KerasModel = Any
Number = Union[int, float]

asarray = np.asarray #type: ignore

def asnumber(x) -> Number:
    if isinstance(x, (int, float)):
        if int(x) == x:
            return int(x)
        return x
    else:
        return asarray(x).item() #type: ignore


class RegularGrid(NamedTuple):
    """Follows the same semantics as ITK.
    https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html"""

    origin: Number = 0
    spacing: Number = 1
    size: int | None = None

    # def __init__(self, origin: float | int = 0, spacing: float | int = 1, size: int | None = None):
    #     self._origin = origin
    #     self._spacing = spacing
    #     self._size = size

    # def __repr__(self):
    #     return f"{self.__class__.__name__}(origin={self._origin}, spacing={self._spacing}, size={self._size})"

    # @property
    # def origin(self) -> float | int:
    #     return self._origin

    # @property
    # def spacing(self) -> float | int:
    #     return self._spacing

    # @property
    # def size(self) -> int | None:
    #     return self._size

    # def _repr_inline_(self, max_width):
    #     return repr(self)

    @property
    def range(self) -> Number:
        return self.spacing * (self.size - 1)  # type: ignore

    @property
    def shape(self) -> tuple[int]:
        if isinstance(self.size, int):
            return (self.size,)

        raise AssertionError

    @property
    def shift(self) -> Number:
        return self.origin % self.spacing

    @property
    def ndim(self) -> int:
        return 1

    @property
    def dtype(self) -> np.dtype:
        if isinstance(self.origin, int) and isinstance(self.spacing, int):
            return np.dtype("int")
        else:
            return np.dtype("float")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.origin == other.origin
                and self.spacing == other.spacing
                and self.size == other.size
            )
        return False

    def __array_function__(self, func, types, args: tuple, kwargs: dict):
        print(func)
        if func in HANDLED_FUNCTIONS:
            try:
                return HANDLED_FUNCTIONS[func](*args, **kwargs)
            except NotImplementedError:
                pass

        args, kwargs = nested_cast_to_array(args, kwargs, dtype=self.__class__)
        return func(*args, **kwargs)

    def __array_ufunc__(self, func, types, args: tuple, kwargs: dict):
        return NotImplemented

    def __array_namespace__(self, *, api_version=None):
        return np

    @classmethod
    def from_coord(cls, coord: Sequence[float], maybe_cast_to_int=False) -> RegularGrid:
        """Assumes coordinate is a regularly spaced sequence of numbers. TODO: add checks and relax this assumption"""
        s = coord[1] - coord[0]
        if maybe_cast_to_int:
            s = int(s) if int(s) == s else s
        return cls(origin=coord[0], spacing=s, size=len(coord))

    def is_aligned(self, other: RegularGrid):
        return (
            other.spacing == self.spacing
            and self.origin % self.spacing == other.origin % other.spacing
        )

    def __array__(self):
        return np.arange(0, self.size) * self.spacing + self.origin  # type: ignore

    def to_numpy(self) -> np.ndarray:
        assert self.size
        return np.arange(self.size) * self.spacing + self.origin

    def __add__(self, other) -> RegularGrid:
        if isinstance(other, self.__class__):
            if other.size == self.size:
                return RegularGrid(
                    self.origin + other.origin,
                    self.spacing + other.spacing,
                    size=self.size,
                )
            else:
                return NotImplemented

        try:
            other = asnumber(other)
            return RegularGrid(
                origin=self.origin + other,  # type: ignore
                spacing=self.spacing,
                size=self.size,
            )
        except:
            pass

        return NotImplemented

    def __sub__(self, other):
        try:
            return RegularGrid(
                origin=self.origin - other, spacing=self.spacing, size=self.size
            )
        except:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __len__(self):
        return self.size

    def __truediv__(self, other):
        try:
            return RegularGrid(
                origin=self.origin / other, spacing=self.spacing / other, size=self.size
            )
        except:
            raise NotImplemented

    def __mul__(self, other):
        try:
            return RegularGrid(
                origin=self.origin * other, spacing=self.spacing * other, size=self.size
            )
        except:
            raise NotImplemented

    def __rmul__(self, other):
        try:
            return RegularGrid(
                origin=other * self.origin, spacing=other * self.spacing, size=self.size
            )
        except:
            raise NotImplemented

    def __getitem__(self, key: int | slice | tuple[int | slice]):
        if isinstance(key, tuple):
            key = key[0]

        if isinstance(key, int):
            if self.size is not None:
                if key < 0:
                    key = self.size - key
                if key >= self.size or key < 0:
                    raise IndexError
            else:
                if key < 0:
                    raise IndexError
            return self.origin + key * self.spacing

        if isinstance(key, slice):  # type: ignore
            if self.size is not None:
                key = slice(*key.indices(self.size))

                origin = self.origin + key.start * self.spacing
                spacing = key.step * self.spacing
                size = max(key.stop - key.start, 0) // key.step
            else:
                start = key.start or 0
                step = key.step or 1

                origin = self.origin + start * self.spacing
                spacing = self.spacing * step
                size = None

            return RegularGrid(origin=origin, spacing=spacing, size=size)

        raise KeyError(key)

    def to_slice(self) -> slice:
        return slice(
            self.origin,
            (self.size + 1) * self.spacing if self.size else None,
            self.spacing,
        )

    def __slice__(self):
        return self.to_slice()

    def downsample(self, shape: DownSamplerShape):
        origin = (
            self.origin
            + (shape.padding + (shape.odd + shape.stride - 1) / 2) * self.spacing
        )
        spacing = self.spacing * shape.stride

        if self.size is None:
            size = self.size
        else:
            size = self.size - shape.padding * 2 - shape.odd
            size = max(size, 0)
            size = size // shape.stride

        return RegularGrid(origin, spacing, size)


class OpenRegularGrid(RegularGrid):
    origin: float | int = 0
    spacing: float | int = 1
    size: None = None

    @property
    def shift(self) -> Number:
        return self.origin % self.spacing

    @property
    def ndim(self) -> int:
        return 1

    @property
    def dtype(self) -> np.dtype:
        if isinstance(self.origin, int) and isinstance(self.spacing, int):
            return np.dtype("int")
        else:
            return np.dtype("float")


class DownSamplerShape(NamedTuple):
    """Stride is the spacing of the ouput elements (in units of input elements).

    spacing of the ouput elements = stride (in units of input elements)
    padding from each side = (padding + odd * 0.5)
    minimum_size = padding * 2 + odd

    Note: odd is either a 0 or 1, and indicates if padd
    """

    stride: int = 1
    padding: int = 0
    odd: int = 0
    shift: int = 0

    def minimum_size(self) -> int:
        return self.padding * 2 + self.odd

    @classmethod
    def from_function(
        cls,
        size_function: Callable[[int], int],
        start: int = 20000,
        max_iter: int = 1000,
    ) -> DownSamplerShape:
        breaks = []
        i = start

        last = size_function(i)
        for _ in range(max_iter):
            if len(breaks) > 1:
                break

            this = size_function(i)
            if this != last:
                breaks.append((i, this))
            last = this
            i += 1

        stride = breaks[1][0] - breaks[0][0]
        min_size = breaks[1][0] - breaks[1][1] * stride
        sign = 1 if min_size < 0 else 1
        pad = min_size // 2 * sign
        odd = min_size % 2 * sign
        return cls(stride=stride, padding=pad, odd=odd)


def keras_model_change_padding(
    model: KerasModel,
    padding: str = "valid",
    copy_weights: bool = True,
) -> KerasModel:
    """
    Convert Keras model into an equivalent model that uses a particular padding (default, "valid.").
    """
    c = model.get_config()
    c = _keras_config_change_padding(c, padding)
    m = model.__class___.from_config(c)
    if copy_weights:
        w = model.get_weights()
        m.set_weights()
    return m


def _keras_config_change_padding(config: dict, padding: str = "valid"):
    """
    Convert Keras model config into an equivalent config that uses a particular padding (default, "valid.")
    """

    if isinstance(config, list):
        return [_keras_config_change_padding(x) for x in config]

    if isinstance(config, tuple):
        return tuple([_keras_config_change_padding(x) for x in config])

    if isinstance(config, dict):
        if "padding" in config:
            config["padding"] = padding

        for k in list(config):
            config[k] = _keras_config_change_padding(config[k])

        return config

    return config


def nested_cast_to_array(*struct, dtype=RegularGrid) -> tuple:
    cache = {}

    def caster(x):
        if isinstance(x, RegularGrid):
            i = id(x)
            if i not in cache:
                cache[i] = asarray(x) #type: ignore
            return cache[i]

    return tree.traverse(caster, struct)  # type: ignore


HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate(X: Sequence, axis=0, out=None, **kwargs):
    if axis != 0:
        raise NotImplemented
    if out is not None:
        raise NotImplemented

    for x in X:
        if not isinstance(x, RegularGrid):
            raise NotImplemented

    if len(X) == 1:
        return X[0]

    i = X[0]

    for x in X[1:]:
        if x.size == 0:
            continue
        if i.size is None:
            raise NotImplementedError
        if not np.isclose(x.spacing, i.spacing): #type: ignore
            raise NotImplementedError
        if not np.isclose(x.origin, i.origin + i.size * i.spacing): #type: ignore
            raise NotImplementedError
        i = RegularGrid(i.origin, i.spacing, i.size + x.size)

    return i
