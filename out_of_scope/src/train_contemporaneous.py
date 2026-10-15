"""Ex-post same-month volatility explanation. This is not a real-time forecast.

Usage: python src/train_contemporaneous.py --period all --scale both
"""

from volatility_analysis_utils import analysis_main


if __name__ == "__main__":
    analysis_main("contemporaneous")
