"""Pure, deterministic domain logic: the quant and ML core of AMME.

Hard rules for everything in this package -- they are what make s.6.1's
byte-for-byte replay test possible:

1. **No I/O.** No network, no disk, no database, no logging side effects.
2. **No wall clock.** Time arrives as an argument or via an injected
   ``amme_shared.clock.Clock``. Nothing calls ``time.time()``.
3. **No hidden randomness.** Every stochastic component takes an explicit
   ``numpy.random.Generator``. No module-level ``np.random`` calls, ever.
4. **No global mutable state.** Components are instantiated, not imported.

The payoff: ``backtest-svc`` composes these classes in-process for a
deterministic replay, while the live services wrap the identical objects in
Redis consumers. There is exactly one implementation of the Avellaneda-Stoikov
maths in this repo, and both paths run it.
"""

from __future__ import annotations

__all__: list[str] = []
