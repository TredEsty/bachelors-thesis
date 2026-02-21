import os
import time
from django.conf import settings
from google import genai
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

#colors for terminal marking
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

class MovieService:
    DATA_DIR = os.path.join(settings.BASE_DIR, 'data')
    CACHE_DIR = os.path.join(settings.BASE_DIR, 'ai_cache') 
    
    os.makedirs(CACHE_DIR, exist_ok=True)

    @staticmethod
    def get_cache(tconst):
        filename = f"{tconst}.html"
        cache_path = os.path.join(MovieService.CACHE_DIR, filename)
        if os.path.exists(cache_path):
            print(f"{GREEN}[CACHE]{RESET} Found cached analysis for '{tconst}'.")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    @staticmethod
    def set_cache(tconst, content):
        filename = f"{tconst}.html"
        cache_path = os.path.join(MovieService.CACHE_DIR, filename)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"{GREEN}[CACHE]{RESET} Analysis saved to cache for '{tconst}'.")

    @staticmethod
    def search_movies(query, limit=10):
        basics_path = os.path.join(MovieService.DATA_DIR, 'title.basics.tsv')
        results = []
        
        if not os.path.exists(basics_path):
            print(f"{RED}[ERROR]{RESET} IMDb TSV files missing.")
            return results

        print(f"{BLUE}[SEARCH]{RESET} Searching for '{query}' in IMDb local database...")
        start_time = time.time()
        
        try:
            for chunk in pd.read_csv(basics_path, sep='\t', low_memory=False, chunksize=100000):
                match = chunk[
                    (chunk['titleType'].isin(['movie', 'tvSeries', 'short'])) & 
                    (chunk['primaryTitle'].str.contains(query, case=False, na=False))
                ]
                
                for _, row in match.iterrows():
                    results.append({
                        'tconst': row['tconst'],
                        'title': row['primaryTitle'],
                        'year': row['startYear']
                    })
                    
                    if len(results) >= limit:
                        elapsed = time.time() - start_time
                        print(f"{GREEN}[FOUND]{RESET} Found {len(results)} matches in {elapsed:.2f}s")
                        return results
                        
            return results

        except Exception as e:
            print(f"{RED}[DATA ERROR]{RESET} {e}")
            return []

    @staticmethod
    def get_movie_data(tconst):
        basics_path = os.path.join(MovieService.DATA_DIR, 'title.basics.tsv')
        ratings_path = os.path.join(MovieService.DATA_DIR, 'title.ratings.tsv')
        
        if not os.path.exists(basics_path) or not os.path.exists(ratings_path):
            return None

        try:
            print(f"{BLUE}[SEARCH]{RESET} Fetching exact data for ID: {tconst}...")
            movie_basics = None
            movie_ratings = {'averageRating': 'N/A', 'numVotes': 0}
            
            for chunk in pd.read_csv(basics_path, sep='\t', low_memory=False, chunksize=100000):
                match = chunk[chunk['tconst'] == tconst]
                if not match.empty:
                    movie_basics = match.iloc[0].to_dict()
                    break
            
            if not movie_basics:
                return None

            for chunk in pd.read_csv(ratings_path, sep='\t', low_memory=False, chunksize=100000):
                match = chunk[chunk['tconst'] == tconst]
                if not match.empty:
                    movie_ratings = match.iloc[0].to_dict()
                    break

            return {
                'primaryTitle': movie_basics['primaryTitle'],
                'startYear': movie_basics['startYear'],
                'rating': movie_ratings,
                'people': []
            }

        except Exception as e:
            print(f"{RED}[DATA ERROR]{RESET} {e}")
            return None

    @staticmethod
    def analyze_movie(tconst):
        #check cache
        cached_result = MovieService.get_cache(tconst)
        if cached_result:
            return cached_result

        #get local data by id
        movie_data = MovieService.get_movie_data(tconst)
        if not movie_data: 
            return "<div>Error: Movie data not found in local dataset.</div>"

        #ai processing
        print(f"{YELLOW}[AI]{RESET} Sending data to Gemini for analysis...")
        
        key_people = [p.get('primaryName', 'Unknown') for p in movie_data.get('people', [])[:5]] 
        prompt = f"""
            Analyze the following title with a focus on its creative pedigree:
            Title: {movie_data.get('primaryTitle')} ({movie_data.get('startYear')})
            Rating: {movie_data.get('rating', {}).get('averageRating', 'N/A')}
            Key Cast: {', '.join(key_people) if key_people else 'Unknown'}
            Key Crew (Director/Writers): {', '.join(key_people) if key_people else 'Unknown'}

            Please provide:
            1. **Description**: A concise summary of the premise.
            2. **The Pedigree**: Identify 2-3 standout previous works from the director and lead cast. Mention if this team has collaborated before or if this genre is their "sweet spot."
            3. **The Verdict**: Based on the rating and the "track record" of the creators involved, is this a must-watch or a skip? 
            4. **The Audience**: Who is this specifically for? (e.g., "Fans of slow-burn noir" or "Casual weekend viewers").
            5. **3 Similar Works**: Recommend titles that share a similar tone or stylistic 'feel'.

            Format in clean, minimal HTML using only <h3>, <p>, and <ul> tags.
            """

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            
            result_html = response.text
            print(f"{YELLOW}[AI]{RESET} Analysis generated successfully.")
            
            #save cache
            MovieService.set_cache(tconst, result_html)
            return result_html

        except Exception as e:
            if "429" in str(e):
                print(f"{RED}[AI ERROR]{RESET} Quota exceeded (Rate limit).")
                return "<div>Quota Reached. Try again in 60s.</div>"
            
            print(f"{RED}[AI ERROR]{RESET} {str(e)}")
            return f"<div>Error: {str(e)}</div>"