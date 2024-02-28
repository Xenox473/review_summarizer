from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class ReviewScraper:
  def __init__(self, url):
    self.url = url
    self.driver = self.get_driver()
    self.driver.get(url)
    load_more_button = self.driver.find_element("id", "load-more-reviews--button")
    self.driver.execute_script("arguments[0].click();", load_more_button)

    time.sleep(2)
    self.soup = BeautifulSoup(self.driver.page_source, 'html.parser')
    self.driver.quit()

  def get_driver(self):
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

  def get_reviews(self):
    album_reviews = self.soup.find(id="album-reviews")
    reviews_df = pd.DataFrame(columns=["review", "score"])
    reviews = zip(album_reviews.find_all(["h6", "div"], class_="card-title h5 font-weight-bold"), album_reviews.find_all("p", class_="card-text album--review--text"))
    for score, review in reviews:
        reviews_df = pd.concat([reviews_df, pd.DataFrame([[review.text, score.text]], columns=["review", "score"])])

    reviews_df.reset_index(drop=True, inplace=True)
    reviews_df.to_csv("./data/" + self.url.split("/")[-1] + "_reviews.csv", index=False)

    return reviews_df
  

if __name__ == "__main__":
  URL = "https://1001albumsgenerator.com/albums/1lPoRKSgZHQAYXxzBsOQ7v/bringing-it-all-back-home"
  scraper = ReviewScraper(URL)
  reviews_df = scraper.get_reviews()
  print(reviews_df)