import requests
from bs4 import BeautifulSoup
import json

def get_nasdaq100_companies():
    url = "https://www.slickcharts.com/nasdaq100"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'table'})
    if table is None:
        print("Error: Table not found on the webpage.")
        return {}

    nasdaq100_dict = {}
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        company_name = cols[1].text.strip()
        ticker = cols[2].text.strip()
        nasdaq100_dict[company_name] = ticker

    return nasdaq100_dict

# Example usage
nasdaq100_companies_dict = get_nasdaq100_companies()

# Save the dictionary to a JSON file
with open('nasdaq100_companies.json', 'w') as json_file:
    json.dump(nasdaq100_companies_dict, json_file, indent=4)

print("NASDAQ-100 companies have been saved to nasdaq100_companies.json")
