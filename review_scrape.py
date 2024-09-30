import os
import serpapi
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API key and SerpApi client
api_key = os.getenv('SERP_API')
client = serpapi.Client(api_key=api_key)

# Function to get reviews from a page
def get_reviews(place_id, next_page_token=None):
    params = {
        'engine': 'google_maps_reviews',
        'type': 'search',
        'place_id': place_id,
        'api_key': api_key,
        'sort_by': 'newest'  # Sort reviews by newest first
    }

    if next_page_token:
        params['next_page_token'] = next_page_token

    return client.search(params)

# Initialize variables
place_id = "ChIJnSvu1sIIYUgRbb4r_ffVLPw"
next_page_token = None
extracted_reviews = []

# Fetch up to 12 pages of results
for _ in range(50):
    try:
        results = get_reviews(place_id, next_page_token)

        # Extract 'iso_date' and 'snippet' from each review
        for review in results.get('reviews', []):
            extracted_reviews.append({
                'Date': review.get('iso_date'),
                'Review': review.get('snippet')
            })

        # Check for next page token
        next_page_token = results.get('serpapi_pagination', {}).get('next_page_token')

        # If no next page token, break the loop
        if not next_page_token:
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Convert the extracted data into a DataFrame
reviews_df = pd.DataFrame(extracted_reviews, columns=['Date', 'Review'])
print(reviews_df)

reviews_df.to_csv('reviews.csv')
