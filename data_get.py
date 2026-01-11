import requests
import json

def get_json_from_url(url):
    """
    Fetch JSON data from a URL.
    
    Args:
        url (str): The URL to fetch JSON from
        
    Returns:
        dict: Parsed JSON data
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    return response.json()

if __name__ == "__main__":
    url = "https://services.swpc.noaa.gov/products/alerts.json"
    data = get_json_from_url(url)
    with open("alerts.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print("Data saved to alerts.json")