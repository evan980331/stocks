from twstock.sentiment import news_rss


def test_score_text_positive():
    assert news_rss._score_text("台積電營收創新高，法說利多") > 0


def test_score_text_negative():
    assert news_rss._score_text("個股重挫，外資賣超，遭砍目標價") < 0


def test_score_text_neutral():
    assert news_rss._score_text("你好") == 0


def test_decode_google_link():
    link = "https://news.google.com/rss/articles/abc?url=https%3A%2F%2Fexample.com%2Fx"
    assert news_rss._decode_google_link(link) == "https://example.com/x"
