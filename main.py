import numpy as np
import pandas as pd
import ollama as ollama
import time
from rich import print
import matplotlib.pyplot as plt
from scraper import scrape
from sentiment import ai_detect
import sys
count = 0
def run():
    scrape(user_input, 'ytdownload.csv')
    dataraw = pd.read_csv('ytdownload.csv', encoding='utf-8')
    #mask = dataraw.apply(lambda row: len(''.join(row.fillna('').astype(str))), axis=1) > 2
    datacleaned = dataraw.to_numpy(dtype=object)
    start_time = time.time()
    spef_movie = datacleaned
    indices = np.where(spef_movie)[0]
    indeces = indices.reshape(-1, 1)
    total_sentiment = 0
    negative_sentamitent = 0
    positive_sentiment = 0
    neutral_sentiment = 0
    labels = ['Negative', 'Neutral', 'Positive']
    #iterating over the indeces
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
    print(f"---")
    print(f"Processing completed in {(end_time - start_time)/60:.2f} minutes")
    print(f"Average score: {average:.4f}")

    #plotting the results
    plt.pie(sizes, labels=labels)
    plt.axis('equal')
    plt.show()
    count += 1 


running = True 
user_input = input("Enter a YouTube video URL to scrape comments from (or 'exit' to quit): ")
while running == True:
    if user_input == 'exit':
       try: 
            sys.exit()
       except SystemExit:
            sys.exit()
    elif count == 1 :
        print("You have already processed a video. Please restart the program to process another video.")
        sys.exit()
    else:
        run()



