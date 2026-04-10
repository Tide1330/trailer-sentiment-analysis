from csv import writer
import csv
import numpy as np
import pandas as pd
import ollama as ollama
import re
import time
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from rich.progress import Progress   
from rich import print
import matplotlib.pyplot as plt

def scrape(url, targetfile):
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
        print(f"Scraping comments from {url} to {targetfile}...")
        with Progress() as progress:
            task = progress.add_task("[red]Scraping...", total=100)
            with open(targetfile, 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'comment_text'])
                start_time = time.time()
                for idx, comment in enumerate(comments):
                        progress.update(task, advance=1)
                        writer.writerow([idx, comment['text'].replace('\n', ' ')])
                        #this is to make sure this program doesnt run for too long(if u have bad wifi)
                        if time.time() - start_time > 500:
                         break
                        if idx >= 100:
                         break
                        #yt was getting rate limited, so added a sleep every 10 comments to avoid that :)
                        if idx % 10 == 0:
                             time.sleep(0.1)
                             continue
                        
        print(f"Scraping completed. Comments saved to {targetfile}.")             
        return targetfile
scrape('https://www.youtube.com/watch?v=kJQP7kiw5Fk', 'ytdownload.csv')
dataraw = pd.read_csv('ytdownload.csv', encoding='utf-8')
mask = dataraw.apply(lambda row: len(''.join(row.fillna('').astype(str))), axis=1) > 2
datacleaned = dataraw.to_numpy(dtype=object)

def ai_detect(comment):
    response = ollama.chat(model='gemma3:1b', messages=[
        {
            'role': 'system',
            'content': 'You are a sentiment analyzer trying to figure out the audiences responses for the movie based on this trailer. Respond with only one of these NUMBERS: Positive: 1, Negative: -1, or Neutral: 0.'
        },
        {
            'role': 'user',
            'content': f"Analyze this Comment: {comment}",
        },
    ])
    ai_text = str(response['message']['content'])
    match = re.search(r'(-1|0|1)', ai_text)
    if match:
        return match.group(0)
    else:
        return 0

start_time = time.time()

spef_movie = datacleaned
indices = np.where(spef_movie)[0]
indeces = indices.reshape(-1, 1)
total_sentiment = 0
negative_sentamitent = 0
positive_sentiment = 0
neutral_sentiment = 0
labels = ['Negative', 'Neutral', 'Positive']
for idx in indices:
    try:
        comment = str(datacleaned[idx, 1])
        sentiment = ai_detect(comment)
        datacleaned[idx, 1] = sentiment
        
        total_sentiment += float(sentiment)
        if sentiment == '1':
            print(f"[bold green]Processed index {idx} | Sentiment: {sentiment}[/bold green]")
            positive_sentiment += 1
        elif sentiment == '-1':
            print(f"[bold red]Processed index {idx} | Sentiment: {sentiment}[/bold red]")
            negative_sentamitent += 1
        else:  
            print(f"[bold white]Processed index {idx} | Sentiment: {sentiment}[/bold white]")
            neutral_sentiment += 1
        
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        datacleaned[idx, 1] = 0
        continue
end_time = time.time()
sizes = [negative_sentamitent, neutral_sentiment, positive_sentiment]
average = total_sentiment / len(indices)
plt.pie(sizes, labels=labels)
plt.axis('equal')
plt.show()

print(f"---")
print(f"Processing completed in {(end_time - start_time)/60:.2f} minutes")
print(f"Average score: {average:.4f}")
