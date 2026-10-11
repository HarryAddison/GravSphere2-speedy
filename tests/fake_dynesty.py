"""A minimal stand-in for ``dynesty`` so the sampler wiring can be tested offline."""

import pickle
import types

import numpy as np

CALLS = []


class FakeResults:
    def __init__(self, samples):
        self._samples = samples

    def samples_equal(self):
        return self._samples


class FakeDynamicNestedSampler:
    def __init__(self, loglike, ptform, ndim, **kwargs):
        self.loglike, self.ptform, self.ndim, self.kwargs = loglike, ptform, ndim, kwargs
        self.run_kwargs = None
        CALLS.append(("init", kwargs))

    @classmethod
    def restore(cls, path, pool=None):
        obj = cls(None, lambda u: u, 3, restored_from=path, pool=pool)
        return obj

    def run_nested(self, **kwargs):
        self.run_kwargs = kwargs
        CALLS.append(("run", kwargs))
        with open(kwargs["checkpoint_file"], "wb") as handle:
            pickle.dump({"fake": True}, handle)

    @property
    def results(self):
        rng = np.random.default_rng(0)
        draws = np.array([self.ptform(rng.uniform(0.4, 0.6, self.ndim)) for _ in range(6)])
        return FakeResults(draws)


def make_module():
    mod = types.ModuleType("dynesty")
    mod.DynamicNestedSampler = FakeDynamicNestedSampler
    return mod
