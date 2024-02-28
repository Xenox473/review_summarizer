from review_scraper import ReviewScraper
from summarizer import Summarizer
import pandas as pd

import streamlit as st

st.title("1001 Albums Generator Review Summarizer")

def submit():
    st.session_state.url = st.session_state.widget
    st.session_state.widget = ''

st.text_input(label="Paste url for album", key='widget', on_change=submit)

if "url" in st.session_state:
  url = st.session_state.url
  st.session_state.url = ""
else:
  url = ""

reviews_df = pd.DataFrame()

if len(url) > 0:
  album_name = url.split("/")[-1]
  st.write(f"Fetching reviews for {album_name}")
  scraper = ReviewScraper(url)
  reviews_df = scraper.get_reviews()

if len(reviews_df) > 0:
  st.dataframe(reviews_df)
  st.button(label = "Generate Summary", on_click=lambda: summarize(album_name))
  
container = st.container()
container.empty()

def summarize(album_name):
  summarizer = Summarizer(album_name)
  summarizer.preprocess()
  summarizer.cluster()
  summary = summarizer.generate_summary()
  container.write(summary)
