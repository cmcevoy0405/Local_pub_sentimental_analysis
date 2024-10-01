<h1 align="center" style="color:#FFFFFF; font-family: 'Arial', sans-serif;">Cutters Wharf Sentiment Analysis 🍺</h1> <p align="left" style="color:#FFFFFF; font-family: 'Verdana', sans-serif;"> <b> Key findings: The pub seemed to be performing well in all areas queried. Customers appeared to have strong opinions, as there was a low number of neutral reviews across all areas. The food had an overall positive sentiment; however, there were still a reasonable number of negative reviews. Price and service performed similarly to food, though slightly less positive. Location and atmosphere were overwhelmingly positive, and management also received significantly positive feedback.</b>

### Table of contents
- [Business Problem](#business-problem)
- [Introduction](#introduction)
- [Scraping application](#scraping-application)
- [Model](#model)
- [Insights](#insights)
- [Limitations](#limitations)

### Business Problem
By using a pre-trained sentiment analysis model and embeddings through SentenceTransformers, it categorizes reviews into negative, neutral, or positive sentiment. The dynamic query system enables users to focus on specific elements of the customer experience (e.g., food quality, service efficiency). The results are visualized in easy-to-understand bar charts, allowing business owners or managers to quickly identify areas needing improvement or reinforcement. This helps guide actionable decisions to enhance customer satisfaction, pricing strategies, and overall service quality.

### Introduction
I have spent a lot of time at this pub and have seen it transform again and again. I've always had my own opinions about the pub and was curious to see if the public shared them. To explore this, I created a Google reviews scraper to extract nearly 500 reviews. Using a pre-trained RoBERTa-base model combined with the querying capabilities of LanceDB, I conducted aspect-based sentiment analysis on the pub to get a deeper understanding of customer feedback. 

### Scraping application
I used Google's SERPAPI to gather data from 35 pages of reviews, although I ideally wanted over 50 pages. However, I was limited to 4 free scrapes using SERPAPI. I saved the reviews data as a CSV file in my directory for easy access and further analysis.

### Model
In this project, I selected the cardiffnlp/twitter-roberta-base-sentiment model, which is based on the RoBERTa architecture and fine-tuned for sentiment analysis on Twitter data, making it suited to my task. I also allowed flexibility in model selection, providing the option to switch between models if needed. The model and tokenizer were integrated into a sentiment analysis pipeline, classifying reviews into negative, neutral, or positive sentiment categories. Additionally, I used the all-MiniLM-L6-v2 model from SentenceTransformers to generate embeddings, enabling deeper semantic searches on the review data.

For storage and retrieval of these embeddings, I utilized LanceDB, a vector database designed for fast similarity search. LanceDB allowed me to efficiently store the review embeddings and retrieve them based on user queries. This design ensured that I could perform semantic searches, retrieving relevant reviews even when specific aspects (e.g., food or service) were inputted. The combination of LanceDB and pre-trained models allowed for both sentiment classification and query-based insights, roviding a useful way to analyze customer feedback and identify actionable insights for improving business performance.

### Insights
The food presented an overall positive sentiment, with 75% of customers agreeing that the food is good. However, 1 in every 4 customers had a negative sentiment about the food, which is still significant. Over a quarter of the customers believed that the food and drinks were overpriced for what they were getting. Service followed the same trend, with more than 25% of customers expressing a negative sentiment about staff service. The atmosphere received a very strong positive sentiment, with 95% of customers believing that the location is great. Management also performed well, with an 80% positive sentiment score.

Based on these insghts I would recommend:

- Staff training should be prioritized. Focus on improving communication, friendliness, and speed of service. You might also consider implementing a feedback mechanism for customers to immediately report poor service experiences, allowing management to address issues in real time.
  
- Consider gathering more detailed feedback from these customers to identify specific areas for improvement, such as taste, presentation, or variety. Offering seasonal specials or refining the menu could address these concerns.
  
- Explore two pricing strategies: either adjust pricing to better align with customer expectations or enhance the perceived value by improving portion sizes, ingredients, or offering meal deals.
  
- The location should be leverged to attract additional customers in, outdoor seating should be as comfortable as possible with the intoduction of balnkets/heat lamps to keep customers warm during winter months.

### Limitations
The reviews range from 2016 all the way to 2024. An improvement for this model would be to split the data by dates, allowing users to examine how the pub has evolved over time. This would enable management to identify which areas have improved and which have worsened, allowing them to reintroduce successful products or methods from the past and evaluate the positive impact of recent changes.
