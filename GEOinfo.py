import requests
import os
import time

class PexelsCartoonScraper:
    def __init__(self, api_key, save_dir="pexels_cartoons"):
        """
        Initialize Pexels scraper with API key
        Get free API key from: https://www.pexels.com/api/new/
        """
        self.api_key = api_key
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.base_url = "https://api.pexels.com/v1/search"
        self.headers = {
            "Authorization": api_key
        }
    
    def search_cartoon_images(self, query="cartoon", per_page=20, page=1):
        """
        Search for cartoon images on Pexels
        """
        params = {
            "query": query,
            "per_page": per_page,
            "page": page
        }
        
        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("photos", [])
            else:
                print(f"API Error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def download_from_pexels(self, query="cartoon", num_images=30):
        """
        Download images from Pexels
        """
        downloaded = []
        page = 1
        per_page = min(80, num_images)  # Max 80 per page
        
        while len(downloaded) < num_images:
            photos = self.search_cartoon_images(query, per_page, page)
            
            if not photos:
                break
            
            for photo in photos:
                if len(downloaded) >= num_images:
                    break
                
                # Get the medium size image
                img_url = photo.get("src", {}).get("medium", "")
                
                if img_url:
                    try:
                        img_response = requests.get(img_url, timeout=10)
                        
                        # Create filename
                        filename = f"cartoon_{photo['id']}.jpg"
                        filepath = os.path.join(self.save_dir, filename)
                        
                        # Save image
                        with open(filepath, 'wb') as f:
                            f.write(img_response.content)
                        
                        downloaded.append(filepath)
                        print(f"Downloaded: {filename}")
                        
                        # Respect rate limits
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"Error downloading image: {e}")
                        continue
            
            page += 1
            time.sleep(1)  # Be respectful to the API
        
        return downloaded

# Usage example for Pexels
def use_pexels():
    # Replace with your actual Pexels API key
    API_KEY = "YOUR_PEXELS_API_KEY_HERE"
    
    if API_KEY == "YOUR_PEXELS_API_KEY_HERE":
        print("Please get a free API key from: https://www.pexels.com/api/new/")
        return
    
    scraper = PexelsCartoonScraper(API_KEY)
    
    # Search for different cartoon styles
    queries = [
        "cartoon character",
        "animation",
        "cartoon background",
        "cartoon illustration",
        "funny cartoon"
    ]
    
    all_downloaded = []
    
    for query in queries:
        print(f"\nSearching for: {query}")
        downloaded = scraper.download_from_pexels(query, num_images=10)
        all_downloaded.extend(downloaded)
        time.sleep(2)
    
    print(f"\nTotal downloaded: {len(all_downloaded)} images")
