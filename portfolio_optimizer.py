# portfolio_optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.05):
        """
        Parameters:
        expected_returns (pd.Series): Expected returns for assets
        cov_matrix (pd.DataFrame): Covariance matrix of returns
        risk_free_rate (float): Risk-free rate (default: 0.05)
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        
    def optimize(self, risk_tolerance):
        """Optimize portfolio based on risk tolerance (1-10 scale)"""
        # Convert risk tolerance to mathematical parameters
        risk_aversion = self._map_risk_tolerance(risk_tolerance)
        
        # Define optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize using Quadratic Programming
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
        """Portfolio optimization objective function"""
        port_return = np.dot(weights, self.expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return - (port_return - risk_aversion * port_volatility)
    
    def _map_risk_tolerance(self, rt):
        """Map 1-10 risk tolerance to optimization parameters"""
        return {
            1: 2.0,   # Very conservative
            2: 1.5,
            3: 1.0,
            4: 0.8,
            5: 0.6,    # Moderate
            6: 0.4,
            7: 0.3,    # Aggressive
            8: 0.2,
            9: 0.1,
            10: 0.05   # Very aggressive
        }.get(rt, 0.6)  # Default to moderate
    
    def _format_results(self, weights):
        """Format optimization results for Streamlit"""
        allocations = []
        for i, ticker in enumerate(self.expected_returns.index):
            allocations.append({
                'ticker': ticker,
                'percentage': weights[i] * 100,
                'expected_return': self.expected_returns.iloc[i],
                'volatility': np.sqrt(self.cov_matrix.iloc[i,i])
            })
        return sorted(allocations, key=lambda x: x['percentage'], reverse=True)

# Example usage for testing
if __name__ == "__main__":
    # Sample data
    er = pd.Series([0.12, 0.15, 0.08], index=['STOCK_A', 'STOCK_B', 'STOCK_C'])
    cov = pd.DataFrame({
        'STOCK_A': [0.04, 0.002, 0.001],
        'STOCK_B': [0.002, 0.09, 0.003],
        'STOCK_C': [0.001, 0.003, 0.16]
    }, index=er.index)
    
    po = PortfolioOptimizer(er, cov)
    print(po.optimize(7))