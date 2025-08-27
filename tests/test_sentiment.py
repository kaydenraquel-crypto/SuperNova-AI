import pytest
import math
from supernova.sentiment import (
    score_text, SentimentResult, 
    fetch_recent_from_x, fetch_recent_from_reddit,
    FIN_LEX, MENTION_BOOST, PUBLIC_FIGS
)


class TestSentimentResult:
    def test_sentiment_result_dataclass(self):
        """Test SentimentResult dataclass functionality."""
        result = SentimentResult(
            score=0.5,
            tokens=["bullish", "earnings", "beat"],
            figures=["buffett", "powell"]
        )
        
        assert result.score == 0.5
        assert result.tokens == ["bullish", "earnings", "beat"]
        assert result.figures == ["buffett", "powell"]
        
    def test_sentiment_result_empty(self):
        """Test SentimentResult with empty data."""
        result = SentimentResult(score=0.0, tokens=[], figures=[])
        
        assert result.score == 0.0
        assert result.tokens == []
        assert result.figures == []


class TestScoreText:
    def test_positive_sentiment_words(self):
        """Test scoring with positive sentiment words."""
        text = "Company beats earnings expectations with bullish outlook"
        result = score_text(text)
        
        assert result.score > 0
        assert "beats" in result.tokens
        assert "bullish" in result.tokens
        assert result.figures == []
        
        # Should include relevant tokens
        assert len(result.tokens) > 0

    def test_negative_sentiment_words(self):
        """Test scoring with negative sentiment words."""
        text = "Stock plunges after earnings miss and downgrade"
        result = score_text(text)
        
        assert result.score < 0
        assert "plunges" in result.tokens or "plunge" in result.tokens
        assert "miss" in result.tokens
        assert "downgrade" in result.tokens

    def test_mixed_sentiment(self):
        """Test scoring with mixed positive and negative words."""
        text = "Company beats revenue but faces investigation and lawsuit concerns"
        result = score_text(text)
        
        # Mixed sentiment - exact score depends on word weights
        assert isinstance(result.score, float)
        assert -1 <= result.score <= 1  # Should be within tanh bounds
        
        # Should contain both positive and negative words
        assert "beats" in result.tokens or "beat" in result.tokens
        assert "investigation" in result.tokens
        assert "lawsuit" in result.tokens

    def test_neutral_text(self):
        """Test scoring with neutral text (no financial keywords)."""
        text = "The weather today is quite pleasant and sunny"
        result = score_text(text)
        
        assert result.score == 0.0  # No financial keywords
        assert result.figures == []
        assert len(result.tokens) > 0  # Should still extract tokens

    def test_public_figure_mentions(self):
        """Test scoring with public figure mentions."""
        text = "Warren Buffett and Jerome Powell discuss market outlook"
        result = score_text(text)
        
        # Should get boost from figure mentions
        assert result.score > 0
        assert "buffett" in result.figures
        assert "powell" in result.figures
        assert len(result.figures) == 2

    def test_multiple_figure_mentions(self):
        """Test scoring with multiple public figure mentions."""
        text = "Buffett, Powell, Yellen, Musk, and Dimon all commented on the market"
        result = score_text(text)
        
        # Should get boost from multiple figures
        expected_boost = MENTION_BOOST * 5  # 5 figures mentioned
        assert result.score >= expected_boost * 0.9  # Allow for tanh compression
        
        expected_figures = ["buffett", "powell", "yellen", "musk", "dimon"]
        for fig in expected_figures:
            assert fig in result.figures

    def test_case_insensitivity(self):
        """Test that scoring is case insensitive."""
        text_lower = "bullish beat upgrade"
        text_upper = "BULLISH BEAT UPGRADE"
        text_mixed = "Bullish Beat Upgrade"
        
        result_lower = score_text(text_lower)
        result_upper = score_text(text_upper)
        result_mixed = score_text(text_mixed)
        
        # All should have same score
        assert result_lower.score == result_upper.score == result_mixed.score

    def test_empty_text(self):
        """Test scoring with empty text."""
        result = score_text("")
        
        assert result.score == 0.0
        assert result.tokens == []
        assert result.figures == []

    def test_whitespace_only(self):
        """Test scoring with whitespace-only text."""
        result = score_text("   \n\t   ")
        
        assert result.score == 0.0
        assert result.tokens == []
        assert result.figures == []

    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly."""
        text = "Company beats! Earnings surge, but lawsuit concerns..."
        result = score_text(text)
        
        # Punctuation should be stripped, words should be recognized
        assert "beats" in result.tokens or "beat" in result.tokens
        assert "surge" in result.tokens
        assert "lawsuit" in result.tokens
        assert result.score != 0  # Should have non-zero score

    def test_contractions_handling(self):
        """Test handling of contractions."""
        text = "Company doesn't miss earnings, it's a beat!"
        result = score_text(text)
        
        # Contractions should be handled
        assert "doesn" in result.tokens or "doesn't" in result.tokens
        assert "beat" in result.tokens

    def test_tanh_normalization(self):
        """Test that scores are normalized using tanh."""
        # Create text with many positive words
        positive_words = ["beat", "bullish", "upgrade", "surge", "record"] * 10
        text = " ".join(positive_words)
        
        result = score_text(text)
        
        # Score should be bounded by tanh function
        assert -1 <= result.score <= 1
        assert result.score > 0.9  # Should be close to 1 for many positive words

    def test_specific_financial_terms(self):
        """Test scoring of specific financial terms."""
        test_cases = [
            ("beat expectations", "beat", True),
            ("earnings miss", "miss", False),
            ("bullish outlook", "bullish", True),
            ("bearish sentiment", "bearish", False),
            ("stock buyback", "buyback", True),
            ("investigation launched", "investigation", False),
            ("guidance raised", "guidance", True),
            ("lawsuit filed", "lawsuit", False),
            ("record profits", "record", True),
            ("price surge", "surge", True),
            ("market plunge", "plunge", False),
            ("analyst downgrade", "downgrade", False),
            ("rating upgrade", "upgrade", True),
        ]
        
        for text, expected_word, should_be_positive in test_cases:
            result = score_text(text)
            
            assert expected_word in result.tokens
            if should_be_positive:
                assert result.score > 0, f"Expected positive score for '{text}'"
            else:
                assert result.score < 0, f"Expected negative score for '{text}'"

    def test_compound_scoring(self):
        """Test scoring with compound sentences."""
        text = "Despite the lawsuit concerns, the company beat earnings and announced a massive buyback program"
        result = score_text(text)
        
        # Should contain both positive and negative elements
        assert "lawsuit" in result.tokens
        assert "beat" in result.tokens
        assert "buyback" in result.tokens
        
        # Net sentiment might be positive due to stronger positive words
        # But this depends on the exact weights in FIN_LEX

    def test_figure_name_variations(self):
        """Test recognition of figure name variations."""
        test_cases = [
            "Warren Buffett",
            "buffett",
            "BUFFETT",
            "Jerome Powell discusses rates",
            "Elon Musk tweets",
            "Janet Yellen comments"
        ]
        
        for text in test_cases:
            result = score_text(text)
            
            # Should recognize at least one figure
            assert len(result.figures) >= 1

    def test_non_english_characters(self):
        """Test handling of non-English characters."""
        text = "Company beats earnings! 📈 Stock surges 20%"
        result = score_text(text)
        
        # Should still extract English words
        assert "beats" in result.tokens or "beat" in result.tokens
        assert "surges" in result.tokens or "surge" in result.tokens

    def test_numbers_and_symbols(self):
        """Test handling of numbers and symbols in text."""
        text = "Stock up 15% after Q3 earnings beat of $2.50 vs $2.30 expected"
        result = score_text(text)
        
        # Numbers and symbols should be ignored, words should be extracted
        assert "beat" in result.tokens
        assert "earnings" in result.tokens
        # Numbers should not be in tokens
        assert "15" not in result.tokens
        assert "2.50" not in result.tokens

    def test_score_consistency(self):
        """Test that scoring is consistent across multiple calls."""
        text = "Bullish beat with record surge despite investigation concerns"
        
        results = [score_text(text) for _ in range(5)]
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].score == results[0].score
            assert results[i].tokens == results[0].tokens
            assert results[i].figures == results[0].figures

    def test_extreme_text_lengths(self):
        """Test scoring with very long and very short texts."""
        # Very short text
        short_result = score_text("beat")
        assert short_result.score > 0
        assert "beat" in short_result.tokens
        
        # Very long text
        long_text = " ".join(["bullish beat surge"] * 100)
        long_result = score_text(long_text)
        assert long_result.score > 0
        assert len(long_result.tokens) == 300  # 3 words * 100 repetitions

    def test_financial_lexicon_coverage(self):
        """Test that all words in FIN_LEX are handled correctly."""
        for word, expected_score in FIN_LEX.items():
            result = score_text(word)
            
            assert word in result.tokens
            # Score should have same sign as expected_score
            if expected_score > 0:
                assert result.score > 0
            elif expected_score < 0:
                assert result.score < 0
            else:
                assert result.score == 0

    def test_public_figures_coverage(self):
        """Test that all public figures are recognized."""
        for figure in PUBLIC_FIGS:
            text = f"{figure} comments on market"
            result = score_text(text)
            
            assert figure in result.figures
            assert result.score >= MENTION_BOOST  # Should get at least the mention boost


class TestFetchFunctions:
    def test_fetch_recent_from_x_returns_empty(self):
        """Test that X fetch function returns empty list (placeholder implementation)."""
        result = fetch_recent_from_x("AAPL")
        assert result == []
        
        result = fetch_recent_from_x("TSLA", max_items=100)
        assert result == []

    def test_fetch_recent_from_reddit_returns_empty(self):
        """Test that Reddit fetch function returns empty list (placeholder implementation)."""
        result = fetch_recent_from_reddit("wallstreetbets")
        assert result == []
        
        result = fetch_recent_from_reddit("investing", "AAPL", max_items=50)
        assert result == []

    def test_fetch_functions_with_various_inputs(self):
        """Test fetch functions with various input types."""
        # Test with different query types
        x_queries = ["AAPL", "Bitcoin", "$SPY", "#earnings", ""]
        reddit_queries = ["", "TSLA", "cryptocurrency", "market crash"]
        
        for query in x_queries:
            result = fetch_recent_from_x(query)
            assert isinstance(result, list)
            assert result == []
        
        for query in reddit_queries:
            result = fetch_recent_from_reddit("wallstreetbets", query)
            assert isinstance(result, list)
            assert result == []

    def test_fetch_functions_max_items_parameter(self):
        """Test fetch functions with different max_items values."""
        max_items_values = [1, 10, 50, 100, 1000]
        
        for max_items in max_items_values:
            x_result = fetch_recent_from_x("AAPL", max_items=max_items)
            reddit_result = fetch_recent_from_reddit("investing", max_items=max_items)
            
            assert isinstance(x_result, list)
            assert isinstance(reddit_result, list)
            assert x_result == []
            assert reddit_result == []


class TestSentimentConstants:
    def test_fin_lex_structure(self):
        """Test that FIN_LEX has expected structure."""
        assert isinstance(FIN_LEX, dict)
        assert len(FIN_LEX) > 0
        
        # All keys should be strings, all values should be numbers
        for word, score in FIN_LEX.items():
            assert isinstance(word, str)
            assert isinstance(score, (int, float))
            assert -1 <= score <= 1  # Reasonable sentiment score range

    def test_public_figs_structure(self):
        """Test that PUBLIC_FIGS has expected structure."""
        assert isinstance(PUBLIC_FIGS, set)
        assert len(PUBLIC_FIGS) > 0
        
        # All entries should be strings
        for fig in PUBLIC_FIGS:
            assert isinstance(fig, str)
            assert fig.islower()  # Should be lowercase for matching

    def test_mention_boost_value(self):
        """Test that MENTION_BOOST has reasonable value."""
        assert isinstance(MENTION_BOOST, (int, float))
        assert 0 < MENTION_BOOST <= 1  # Should be positive but not too large

    def test_lexicon_word_uniqueness(self):
        """Test that financial lexicon words are unique."""
        words = list(FIN_LEX.keys())
        assert len(words) == len(set(words))  # No duplicates

    def test_public_figures_known_entries(self):
        """Test that expected public figures are in the set."""
        expected_figures = ["buffett", "powell", "musk", "yellen"]
        
        for fig in expected_figures:
            assert fig in PUBLIC_FIGS

    def test_lexicon_balanced_sentiment(self):
        """Test that lexicon contains both positive and negative words."""
        positive_words = [word for word, score in FIN_LEX.items() if score > 0]
        negative_words = [word for word, score in FIN_LEX.items() if score < 0]
        
        assert len(positive_words) > 0
        assert len(negative_words) > 0
        
        # Should have reasonable balance
        total_words = len(FIN_LEX)
        assert len(positive_words) / total_words > 0.2  # At least 20% positive
        assert len(negative_words) / total_words > 0.2  # At least 20% negative


class TestSentimentEdgeCases:
    def test_very_long_text_performance(self):
        """Test performance with very long text."""
        # Create very long text
        long_text = " ".join(["bullish beat surge downgrade miss"] * 1000)
        
        # Should complete without issues
        result = score_text(long_text)
        
        assert isinstance(result.score, float)
        assert isinstance(result.tokens, list)
        assert isinstance(result.figures, list)
        assert len(result.tokens) == 5000  # 5 words * 1000 repetitions

    def test_unicode_text(self):
        """Test handling of Unicode characters."""
        text = "Company beats earnings! 📈 Très bullish! 日本市場 surge"
        result = score_text(text)
        
        # Should extract English words correctly
        assert "beats" in result.tokens or "beat" in result.tokens
        assert "bullish" in result.tokens
        assert "surge" in result.tokens

    def test_html_entities(self):
        """Test handling of HTML entities."""
        text = "Company &quot;beats&quot; earnings &amp; shows bullish outlook"
        result = score_text(text)
        
        # Should handle HTML entities gracefully
        assert "beats" in result.tokens or "beat" in result.tokens
        assert "bullish" in result.tokens

    def test_repeated_figures(self):
        """Test handling of repeated public figure mentions."""
        text = "Buffett says buy, Buffett recommends hold, Buffett suggests patience"
        result = score_text(text)
        
        # Should only count figure once in the figures list
        assert result.figures == ["buffett"]
        # But should get boost for each mention in scoring
        assert result.score >= MENTION_BOOST

    def test_partial_figure_matches(self):
        """Test that partial matches don't count as figure mentions."""
        text = "The buffer zone shows musk scent in the market"
        result = score_text(text)
        
        # "buffer" contains "buffett" and "musk" is there, but contexts are wrong
        # Only "musk" should be recognized as it's a complete word match
        assert "musk" in result.figures
        assert "buffett" not in result.figures

    def test_sentiment_score_bounds(self):
        """Test that sentiment scores stay within expected bounds."""
        # Create extremely positive text
        extreme_positive = " ".join(list(FIN_LEX.keys()) * 10)
        result_pos = score_text(extreme_positive)
        
        # Should be bounded by tanh
        assert -1 <= result_pos.score <= 1
        
        # Create extremely negative text by filtering negative words
        negative_words = [word for word, score in FIN_LEX.items() if score < 0]
        extreme_negative = " ".join(negative_words * 10)
        result_neg = score_text(extreme_negative)
        
        assert -1 <= result_neg.score <= 1
        assert result_neg.score < 0

    def test_malformed_input_handling(self):
        """Test handling of various malformed inputs."""
        malformed_inputs = [
            None,  # This will cause TypeError, but test should handle it
            123,   # Number instead of string
            [],    # List instead of string
            {},    # Dict instead of string
        ]
        
        for bad_input in malformed_inputs:
            try:
                result = score_text(bad_input)
                # If it doesn't crash, result should be reasonable
                assert isinstance(result, SentimentResult)
            except (TypeError, AttributeError):
                # Expected to fail with wrong input types
                pass