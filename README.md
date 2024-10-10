# Album Review Summarizer
## Abstract
Album reviews are a valuable source of information for music consumers. However, they can be long and tedious to read. This project was inspired by browsing the [1001 album generator](https://1001albumsgenerator.com/) website, which has hundreds of "listener notes" for each album, making it difficult to parse through and make sense of. This project introduces a pipeline that summarizes album reviews through extractive and abstractive summarization techniques. The proposed pipeline has several advantages over existing methods for displaying album reviews, including its ability to capture the most important aspects of the reviews, and its potential to be used as a content moderation tool by ignoring extreme or unrelated reviews. The pipeline involves a series of NLP, machine learning, and graph related techniques to generate a summary. It uses parts of speech tagging to focus on the more important types of words, clustering and transformers to group similar documents together and generate summaries from extracted sentences respectively, and a network of document words to implement the position rank algorithm.

## Steps to run files
This repo contains two files: `review_scraper.py` and `summarizer.py`. 

### Step 1: Run `review_scraper.py`

The file can be run using the following command: `python review_scraper.py`. A url for the album you'd like summarized needs to be set as the url variable in the file. The album can be any one of the albums available on the [1001 album generator](https://1001albumsgenerator.com/) website.

### Step 2: Run 'summarizer.py`
Once the data is scraped and stored in a `data` folder, you can run the summarizer file to generate a summary of the reviews. The album name will need to be set in the summarizer.py file.
