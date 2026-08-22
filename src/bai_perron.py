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

    def _best_additional_break(
        self,
        start,
        end
    ):
        """
        Find the best single additional breakpoint within
        an existing regime.

        Parameters
        ----------
        start : int
            Inclusive start index of the regime.

        end : int
            Exclusive end index of the regime.

        Returns
        -------
        tuple
            best_break : int or None
                Location of the breakpoint producing
                the largest RSS reduction.

            rss_no_break : float
                RSS of the unsplit regime.

            rss_with_break : float
                Minimum RSS after introducing one break.

            rss_reduction : float
                Reduction in RSS due to the additional break.
        """

        rss_no_break = self._segment_rss(
            start,
            end
        )

        best_break = None
        best_split_rss = np.inf

        first_candidate = (
            start + self.min_segment
        )

        last_candidate = (
            end - self.min_segment
        )

        for break_idx in range(
            first_candidate,
            last_candidate + 1
        ):

            left_rss = self._segment_rss(
                start,
                break_idx
            )

            right_rss = self._segment_rss(
                break_idx,
                end
            )

            split_rss = (
                left_rss + right_rss
            )

            if split_rss < best_split_rss:
                best_split_rss = split_rss
                best_break = break_idx

        if best_break is None:
            return (
                None,
                rss_no_break,
                np.inf,
                0.0
            )

        rss_reduction = (
            rss_no_break
            - best_split_rss
        )

        return (
            best_break,
            rss_no_break,
            best_split_rss,
            rss_reduction
        )
        
    def _supf_statistic(
        self,
        start,
        end,
        break_idx
    ):
        """
        Compute a supF-style statistic for testing one additional
        structural break inside an existing segment.

        Parameters
        ----------
        start : int
            Inclusive start index.

        end : int
            Exclusive end index.

        break_idx : int
            Candidate breakpoint location.

        Returns
        -------
        float
            F-style test statistic.
        """

        # RSS without an additional break
        rss_restricted = self._segment_rss(
            start,
            end
        )

        # RSS after splitting the segment at break_idx
        rss_left = self._segment_rss(
            start,
            break_idx
        )

        rss_right = self._segment_rss(
            break_idx,
            end
        )

        rss_unrestricted = (
            rss_left + rss_right
        )

        # Number of observations in the current segment
        n_segment = end - start

        # Restricted model: one mean parameter
        k_restricted = 1

        # Unrestricted model: two regime mean parameters
        k_unrestricted = 2

        numerator_df = (
            k_unrestricted
            - k_restricted
        )

        denominator_df = (
            n_segment
            - k_unrestricted
        )

        if denominator_df <= 0:
            return np.nan

        numerator = (
            rss_restricted
            - rss_unrestricted
        ) / numerator_df

        denominator = (
            rss_unrestricted
            / denominator_df
        )

        if denominator <= 0:
            return np.nan

        f_stat = numerator / denominator

        return f_stat

    def _supf_test_segment(
        self,
        start,
        end
    ):
        """
        Search all admissible breakpoint locations within a segment
        and return the maximum F-style statistic.

        Parameters
        ----------
        start : int
            Inclusive start index.

        end : int
            Exclusive end index.

        Returns
        -------
        dict
            Best breakpoint location and corresponding supF statistic.
        """

        first_candidate = (
            start + self.min_segment
        )

        last_candidate = (
            end - self.min_segment
        )

        # Segment too short to contain another admissible break
        if first_candidate > last_candidate:
            return {
                "BestBreak": None,
                "SupF": np.nan
            }

        best_break = None
        best_stat = -np.inf

        for break_idx in range(
            first_candidate,
            last_candidate + 1
        ):

            f_stat = self._supf_statistic(
                start,
                end,
                break_idx
            )

            if np.isnan(f_stat):
                continue

            if f_stat > best_stat:
                best_stat = f_stat
                best_break = break_idx

        return {
            "BestBreak": best_break,
            "SupF": best_stat
        }

    def _supf_monte_carlo_critical_value(
        self,
        start,
        end,
        n_simulations=1000,
        alpha=0.05,
        random_state=42
    ):
        """
        Estimate an empirical critical value for the supF statistic
        under the null hypothesis of no structural break.

        Parameters
        ----------
        start : int
            Inclusive segment start.

        end : int
            Exclusive segment end.

        n_simulations : int
            Number of Monte Carlo simulations.

        alpha : float
            Significance level.

        random_state : int
            Random seed.

        Returns
        -------
        float
            Empirical supF critical value.
        """

        rng = np.random.default_rng(
            random_state
        )

        n_segment = end - start

        segment_sum = (
            self._cum_sum[end]
            - self._cum_sum[start]
        )

        segment_sq_sum = (
            self._cum_sq_sum[end]
            - self._cum_sq_sum[start]
        )

        segment_mean = (
            segment_sum
            / n_segment
        )

        segment_variance = (
            segment_sq_sum
            - (segment_sum ** 2)
            / n_segment
        ) / (n_segment - 1)

        segment_sd = np.sqrt(
            segment_variance
        )

        simulated_supf = []

        for _ in range(
            n_simulations
        ):

            simulated = rng.normal(
                loc=segment_mean,
                scale=segment_sd,
                size=n_segment
            )

            temp_model = BaiPerron(
                max_breaks=self.max_breaks,
                min_segment=self.min_segment
            )

            temp_model._prepare_cumulative_sums(
                simulated
            )

            result = (
                temp_model._supf_test_segment(
                    0,
                    n_segment
                )
            )

            simulated_supf.append(
                result["SupF"]
            )

        critical_value = np.quantile(
            simulated_supf,
            1 - alpha
        )

        return critical_value
