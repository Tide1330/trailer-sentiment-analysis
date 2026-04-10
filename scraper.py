from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from rich.progress import Progress   
from rich import print
import csv
import time

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
                if time.time() - start_time > 500:
                    break
                if idx >= 100:
                    break
                if idx % 10 == 0:
                    time.sleep(0.1)
                    continue
    print(f"Scraping completed. Comments saved to {targetfile}.")
    return targetfile