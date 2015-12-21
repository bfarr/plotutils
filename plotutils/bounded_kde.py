import numpy as np
from scipy.special import erf
from scipy.stats import gaussian_kde

class Bounded_kde(gaussian_kde):
    r"""Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, low=None, high=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param low: The lower domain boundary.

        :param high: The upper domain boundary."""
        pts = np.atleast_1d(pts)

        assert pts.ndim == 1, 'Bounded_kde can only be one-dimensional'
        
        super(Bounded_kde, self).__init__(pts, *args, **kwargs)

        self._low = low
        self._high = high

    @property
    def low(self):
        """The lower bound of the domain."""
        return self._low

    @property
    def high(self):
        """The upper bound of the domain."""
        return self._high

    def evaluate(self, xs):
        """Return an estimate of the density evaluated at the given
        points."""
        xs = np.atleast_1d(xs)
        assert xs.ndim == 1, 'points must be one-dimensional'

        pdf = super(Bounded_kde, self).evaluate(xs)

        if self.low is not None:
            pdf += super(Bounded_kde, self).evaluate(2.0*self.low - xs)

        if self.high is not None:
            pdf += super(Bounded_kde, self).evaluate(2.0*self.high - xs)

        return pdf

    __call__ = evaluate


class Bounded_2d_kde(gaussian_kde):
    r"""Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.

        :param ylow: The lower y domain boundary.

        :param yhigh: The upper y domain boundary.
        """
        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'

        super(Bounded_2d_kde, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    @property
    def ylow(self):
        """The lower bound of the y domain."""
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain."""
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        pdf = super(Bounded_2d_kde, self).evaluate(pts.T)

        if np.any([self.xlow, self.xhigh, self.ylow, self.yhigh]):
            pts_p = pts.copy()

        if self.ylow is not None:
            pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
            pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
            pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]

        if self.yhigh is not None:
            pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]
            pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
            pts_p[:, 1] = 2.0*self.yhigh + pts_p[:, 1]

        if self.xlow is not None:
            pts_p[:, 0] = 2.0*self.xlow - pts[:, 0]
            pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
            if self.ylow is not None:
                pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
                pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
                pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
            if self.yhigh is not None:
                pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]
                pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
                pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]

        if self.xhigh is not None:
            pts_p[:, 0] = 2.0*self.xhigh - pts[:, 0]
            pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
            if self.ylow is not None:
                pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
                pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
                pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
            if self.yhigh is not None:
                pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]
                pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
                pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]

        return pdf

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results
