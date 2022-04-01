from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import gamma as sp_gamma


class RandomVariable(BaseModel):
    position: float = Field(..., description="data position")
    prob: float = Field(..., ge=0, description="probability density/mass at position")


def gamma(
    x: np.ndarray,
    low: Optional[float] = None,
    high: Optional[float] = None,
    shape: float = 1e-3,
    scale: float = 1e3,
) -> List[RandomVariable]:
    """
    Gamma distribution: x ~ (const.) * x^(shape - 1) * exp(- x / scale)
    Distribution is limited on range of [low, high].
    If low(high) is None, low(high) is set as the minimum(maximum) value of x.
    """
    if low is None:
        low = min(x)
    if high is None:
        high = max(x)
    rv = sp_gamma(a=shape, scale=scale)
    norm = rv.cdf(x=high) - rv.cdf(x=low)
    probs = rv.pdf(x=x) / norm
    return [RandomVariable(position=i, prob=p) for (i, p) in zip(x, probs)]


def uniform(x: np.ndarray, low: float = 0.0, high: float = 1.0) -> List[RandomVariable]:
    """
    Uniform distribution: p(x) = 1 / (high - low)
    """
    prob = 1 / (high - low)
    return [RandomVariable(position=position, prob=prob) for position in x]
