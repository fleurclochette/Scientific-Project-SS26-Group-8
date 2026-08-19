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
        """

        y = np.asarray(y)
        n = len(y)

        rss_matrix = np.full((n, n), np.inf)

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
        
    def _dynamic_programming(self, y):
        """
        Find optimal structural breakpoints using
        dynamic programming.

        Parameters
        ----------
        y : array-like
            Time series.

        Returns
        -------
        dict
            Optimal breakpoints and total RSS for
            each possible number of breaks.
        """

        y = np.asarray(y)
        n = len(y)

        # Build the RSS matrix
        rss_matrix = self._build_rss_matrix(y)

        # Maximum number of regimes is maximum breaks + 1
        max_regimes = self.max_breaks + 1

        # Store minimum RSS for each number of regimes
        costs = np.full(
            (max_regimes + 1, n),
            np.inf
        )

        # Store breakpoint locations
        breakpoints = {}

        # One regime: entire series
        costs[1, :] = rss_matrix[0, :]

        # Dynamic programming
        for regimes in range(2, max_regimes + 1):

            for end in range(
                regimes * self.min_segment - 1,
                n
            ):

                best_cost = np.inf
                best_break = None

                # Possible location of the previous breakpoint
                for previous_end in range(
                    (regimes - 1) * self.min_segment - 1,
                    end - self.min_segment + 1
                ):

                    previous_cost = costs[
                        regimes - 1,
                        previous_end
                    ]

                    current_cost = rss_matrix[
                        previous_end + 1,
                        end
                    ]

                    total_cost = (
                        previous_cost +
                        current_cost
                    )

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_break = previous_end

                costs[regimes, end] = best_cost

                breakpoints[
                    (regimes, end)
                ] = best_break

        return {
            "costs": costs,
            "breakpoints": breakpoints
        }
