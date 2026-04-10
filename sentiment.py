import ollama as ollama
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
analyzer = SentimentIntensityAnalyzer()

def ai_detect(comment):
    scores = analyzer.polarity_scores(comment)
    compound = scores['compound']
    if -0.05 < compound < 0.05:
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
    elif compound >= 0.05:
       return 1
    else:
       return -1