import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

# Configure logging (if not already configured elsewhere)
logging.basicConfig(filename='app.log', level=logging.ERROR)

class PortfolioOptimizer:
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.05):
        """
        Initializes the PortfolioOptimizer with expected returns, covariance matrix, and risk-free rate.

        Args:
            expected_returns (pd.Series): Series of expected returns for each asset.
            cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
            risk_free_rate (float): Risk-free rate of return.
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)

    def optimize(self, risk_tolerance):
        """
        Optimizes the portfolio allocations based on the given risk tolerance.

        Args:
            risk_tolerance (int): Risk tolerance level (1-10).

        Returns:
            list: A list of asset allocations with ticker, percentage, expected return, and volatility.
                   Returns an empty list if optimization fails.
        """
        try:
            risk_aversion = self._map_risk_tolerance(risk_tolerance)
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            x0 = np.array([1 / self.n_assets] * self.n_assets)

            result = minimize(
                self._objective_function,
                x0,
                args=(risk_aversion,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                logging.error(f"Optimization failed: {result.message}")
                return []  # Return empty list to indicate failure

            return self._format_results(result.x)

        except Exception as e:
            logging.error(f"Error during portfolio optimization: {e}")
            return []  # Return empty list if an error occurs

    def _objective_function(self, weights, risk_aversion):
        """
        Calculates the objective function (negative Sharpe ratio) to be minimized.

        Args:
            weights (np.array): Array of asset weights in the portfolio.
            risk_aversion (float): Risk aversion coefficient.

        Returns:
            float: The negative Sharpe ratio.
        """
        port_return = np.dot(weights, self.expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return -(port_return - risk_aversion * port_volatility)

    def _map_risk_tolerance(self, rt):
        """
        Maps risk tolerance levels to risk aversion coefficients.

        Args:
            rt (int): Risk tolerance level (1-10).

        Returns:
            float: Risk aversion coefficient.
        """
        # Added a default value to the get method to handle unexpected risk tolerance values
        return {
            1: 2.0, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.6,
            6: 0.4, 7: 0.3, 8: 0.2, 9: 0.1, 10: 0.05
        }.get(rt, 0.6)

    def _format_results(self, weights):
        """
        Formats the optimization results into a list of asset allocations.

        Args:
            weights (np.array): Array of optimized asset weights.

        Returns:
            list: A list of asset allocations with ticker, percentage, expected return, and volatility.
        """
        allocations = []
        for i, ticker in enumerate(self.expected_returns.index):
            allocations.append({
                'ticker': ticker,
                'percentage': weights[i] * 100,
                'expected_return': self.expected_returns.iloc[i],
                'volatility': np.sqrt(self.cov_matrix.iloc[i, i])
            })
        return sorted(allocations, key=lambda x: x['percentage'], reverse=True)