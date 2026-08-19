"""
Bai-Perron multiple structural breakpoint implementation.

This module contains the custom Python implementation used to
identify structural breaks in the GARCH conditional volatility
series of the S&P 500.
"""

import numpy as np
import pandas as pd

class BaiPerron:
    """
    Custom implementation of the Bai-Perron
    multiple structural breakpoint methodology.
    """

    def __init__(self, max_breaks=6, min_segment=60):
        """
        Initialize the Bai-Perron model.

        Parameters
        ----------
        max_breaks : int
            Maximum number of structural breaks to consider.

        min_segment : int
            Minimum number of observations allowed
            in each regime.
        """

        self.max_breaks = max_breaks
        self.min_segment = min_segment

    def _segment_rss(self, y, start, end):
        """
        Calculate the residual sum of squares (RSS)
        for one segment of the time series.

        Parameters
        ----------
        y : array-like
            Time series.

        start : int
            Starting observation index.

        end : int
            Ending observation index.

        Returns
        -------
        float
            Residual sum of squares for the segment.
        """

        segment = np.asarray(y[start:end])

        segment_mean = np.mean(segment)

        rss = np.sum(
            (segment - segment_mean) ** 2
        )

        return rss
        
def _build_rss_matrix(self, y):
    """
    Build a matrix containing the RSS for every
    admissible segment of the time series.

    Parameters
    ----------
    y : array-like
        Time series.

    Returns
    -------
    numpy.ndarray
        Matrix containing segment RSS values.
    """

    y = np.asarray(y)
    n = len(y)

    # Initialize matrix with infinity.
    # Infinity means that the segment is not admissible.
    rss_matrix = np.full((n, n), np.inf)

    # Calculate RSS for every segment that satisfies
    # the minimum segment length.
    for start in range(n):
        for end in range(
            start + self.min_segment,
            n + 1
        ):
            rss_matrix[start, end - 1] = self._segment_rss(
                y,
                start,
                end
            )

    return rss_matrix
