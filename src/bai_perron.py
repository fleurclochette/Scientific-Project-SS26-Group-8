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

    def _segment_rss(self, start, end):
        """
        Calculate RSS for the segment y[start:end]
        using cumulative sums.
        """

        n_segment = end - start

        if n_segment <= 0:
            return np.inf

        segment_sum = (
            self._cum_sum[end]
            - self._cum_sum[start]
        )

        segment_sq_sum = (
            self._cum_sq_sum[end]
            - self._cum_sq_sum[start]
        )

        rss = (
            segment_sq_sum
            - (segment_sum ** 2) / n_segment
        )

        return rss

    def _build_rss_matrix(self, y):
        """
        Build the RSS matrix for every admissible segment.
        """

        y = np.asarray(y, dtype=float)
        n = len(y)
        self._prepare_cumulative_sums(y)

        rss_matrix = np.full(
            (n, n),
            np.inf
        )

        for start in range(n):
            minimum_end = (
                start
                + self.min_segment
            )

            for end in range(
                minimum_end,
                n + 1
            ):
                rss_matrix[
                    start,
                    end - 1
                ] = self._segment_rss(
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
    def _get_breakpoints(self, dp_results, n_breaks, n):
        """
        Reconstruct the optimal breakpoint locations
        from the dynamic programming results.

        Parameters
        ----------
        dp_results : dict
            Results returned by _dynamic_programming().

        n_breaks : int
            Number of structural breaks.

        n : int
            Number of observations.

        Returns
        -------
        list
            Estimated breakpoint locations.
        """

        breakpoints = dp_results["breakpoints"]

        # Number of regimes equals number of breaks + 1
        regimes = n_breaks + 1

        current_end = n - 1

        estimated_breakpoints = []

        # Work backwards through the dynamic programming table
        for regime in range(regimes, 1, -1):

            previous_end = breakpoints[
                (regime, current_end)
            ]

            estimated_breakpoints.append(
                previous_end + 1
            )

            current_end = previous_end

        # Reverse because we reconstructed backwards
        estimated_breakpoints.reverse()

        return estimated_breakpoints
        
    def _calculate_bic(self, rss, n, n_breaks):
        """
        Calculate the Bayesian Information Criterion (BIC)
        for a given number of structural breaks.

        Parameters
        ----------
        rss : float
            Total residual sum of squares.

        n : int
            Number of observations.

        n_breaks : int
            Number of structural breaks.

        Returns
        -------
        float
            BIC value.
        """

        # Number of regimes
        n_regimes = n_breaks + 1

        # For an intercept-only model, each regime
        # has one estimated mean parameter.
        k = n_regimes

        bic = (
            n * np.log(rss / n)
            + k * np.log(n)
        )

        return bic    

    def _prepare_cumulative_sums(self, y):
        """
        Precompute cumulative sums and cumulative squared sums.
        """

        y = np.asarray(y, dtype=float)

        self._cum_sum = np.concatenate([
            [0.0],
            np.cumsum(y)
        ])

        self._cum_sq_sum = np.concatenate([
            [0.0],
            np.cumsum(y ** 2)
        ])
