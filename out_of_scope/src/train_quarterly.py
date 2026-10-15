"""Next-quarter volatility forecasting with an unpenalized AR(2) block.

Usage: python src/train_quarterly.py --period all --scale both
"""

from volatility_analysis_utils import analysis_main


if __name__ == "__main__":
    analysis_main("quarterly")
