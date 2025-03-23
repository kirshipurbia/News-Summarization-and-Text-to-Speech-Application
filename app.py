import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from gtts import gTTS
from googletrans import Translator
import matplotlib.pyplot as plt

# ----------------------------------------
# News Scraping Function
# ----------------------------------------
def scrape_news(company):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://news.google.com/search?q={company}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    for item in soup.find_all('article', limit=10):
        title = item.find('a', class_='JtKRv').text
        link = "https://news.google.com" + item.find('a')['href'].lstrip('.')
        articles.append({"title": title, "link": link})
    return articles

# ----------------------------------------
# Sentiment Analysis Function
# ----------------------------------------
def analyze_sentiment(articles):
    model = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    for article in articles:
        text = article["title"]
        result = model(text)[0]
        rating = int(result['label'].split()[0])
        article["sentiment"] = "POSITIVE" if rating >=4 else "NEUTRAL" if rating==3 else "NEGATIVE"
    return articles

# ----------------------------------------
# Hindi TTS Function
# ----------------------------------------
def text_to_hindi_speech(text, filename="output.mp3"):
    translator = Translator()
    translated = translator.translate(text, dest='hi').text
    tts = gTTS(translated, lang='hi')
    tts.save(filename)
    return filename

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.title("📰 Company News Analyzer")
company = st.text_input("Company ka naam daalo", placeholder="Reliance, Infosys...")

if st.button("Report Generate Karo"):
    articles = scrape_news(company)
    if len(articles) == 0:
        st.error("Koi news nahi mili! Kisi aur company ka naam try karo.")
    else:
        articles = analyze_sentiment(articles)
        
        # Show Articles
        st.subheader("Top 10 News ke Results:")
        for idx, article in enumerate(articles, 1):
            st.write(f"{idx}. **{article['title']}** - _{article['sentiment']}_")
        
        # Pie Chart
        sentiments = [a['sentiment'] for a in articles]
        pos = sentiments.count("POSITIVE")
        neg = sentiments.count("NEGATIVE")
        neu = len(sentiments) - pos - neg
        
        fig, ax = plt.subplots()
        ax.pie([pos, neg, neu], labels=["Positive", "Negative", "Neutral"], autopct="%1.1f%%", colors=["green", "red", "gray"])
        st.pyplot(fig)
        
        # Hindi Audio
        summary = f"{company} ke baare mein {pos} positive, {neg} negative, aur {neu} neutral news mili."
        audio_file = text_to_hindi_speech(summary)
        st.audio(audio_file)