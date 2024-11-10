import unittest
from src.whale_tracking import WhaleTracker
from src.sentiment_analysis import SentimentAnalyzer
from src.strategy_adjustment import StrategyAdjuster
from src.technical_indicators import TechnicalIndicators
from src.risk_management import RiskManager

class TestWhaleMomentumBot(unittest.TestCase):

    def setUp(self):
        self.whale_tracker = WhaleTracker()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.strategy_adjuster = StrategyAdjuster()
        self.technical_indicators = TechnicalIndicators()
        self.risk_manager = RiskManager()

    def test_track_whales(self):
        result = self.whale_tracker.track_whales()
        self.assertIsInstance(result, list)

    def test_analyze_trade_impact(self):
        trade_data = {'amount': 1000, 'price': 50}
        impact = self.whale_tracker.analyze_trade_impact(trade_data)
        self.assertIsInstance(impact, float)

    def test_fetch_sentiment_data(self):
        sentiment_data = self.sentiment_analyzer.fetch_sentiment_data()
        self.assertIsInstance(sentiment_data, dict)

    def test_parse_sentiment(self):
        sentiment_data = {'positive': 0.7, 'negative': 0.3}
        parsed = self.sentiment_analyzer.parse_sentiment(sentiment_data)
        self.assertIsInstance(parsed, float)

    def test_evaluate_performance(self):
        performance = self.strategy_adjuster.evaluate_performance()
        self.assertIsInstance(performance, dict)

    def test_adjust_strategy(self):
        new_strategy = self.strategy_adjuster.adjust_strategy()
        self.assertIsInstance(new_strategy, dict)

    def test_calculate_macd(self):
        prices = [1, 2, 3, 4, 5]
        macd = self.technical_indicators.calculate_macd(prices)
        self.assertIsInstance(macd, float)

    def test_calculate_rsi(self):
        prices = [1, 2, 3, 4, 5]
        rsi = self.technical_indicators.calculate_rsi(prices)
        self.assertIsInstance(rsi, float)

    def test_set_stop_loss(self):
        stop_loss = self.risk_manager.set_stop_loss(100, 0.1)
        self.assertEqual(stop_loss, 90)

    def test_apply_trailing_stop(self):
        trailing_stop = self.risk_manager.apply_trailing_stop(100, 5)
        self.assertEqual(trailing_stop, 95)

if __name__ == '__main__':
    unittest.main()