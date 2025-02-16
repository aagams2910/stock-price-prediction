import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.05):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        
    def optimize(self, risk_tolerance):
        risk_aversion = self._map_risk_tolerance(risk_tolerance)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self._objective_function,
            x0,
            args=(risk_aversion,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return self._format_results(result.x)
    
    def _objective_function(self, weights, risk_aversion):
        port_return = np.dot(weights, self.expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return - (port_return - risk_aversion * port_volatility)
    
    def _map_risk_tolerance(self, rt):
        return {
            1: 2.0, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.6,
            6: 0.4, 7: 0.3, 8: 0.2, 9: 0.1, 10: 0.05
        }.get(rt, 0.6)
    
    def _format_results(self, weights):
        allocations = []
        for i, ticker in enumerate(self.expected_returns.index):
            allocations.append({
                'ticker': ticker,
                'percentage': weights[i] * 100,
                'expected_return': self.expected_returns.iloc[i],
                'volatility': np.sqrt(self.cov_matrix.iloc[i, i])
            })
        return sorted(allocations, key=lambda x: x['percentage'], reverse=True)
