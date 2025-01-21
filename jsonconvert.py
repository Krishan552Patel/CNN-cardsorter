import json
import csv

# Load the JSON file
json_file_path = "card.json"
csv_file_path = "card.csv"

with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract relevant data
cards_data = []

for card in data:
    unique_id = card.get("unique_id", "")
    for printing in card.get("printings", []):
        card_id = printing.get("id", "")
        image_url = printing.get("image_url", "")
        Foiling = printing.get("foiling", "")
        if(Foiling=="S"):
            Foiling = "Standard"
        if(Foiling=="R"):
            Foiling = "Rainbow"
        if(Foiling=="C"):
            Foiling = "Cold Foil"
        if(Foiling=="G"):
            Foiling = "Gold Foil"
        edition = printing.get("edition", "")
        if(edition=="A"):
            edition = "Alpha"
        if(edition=="F"):
            edition = "First Edition"
        if(edition=="U"):
            edition = "Unlimited"
        if(edition=="N"):
            edition=""
            
        cards_data.append([unique_id, card_id, image_url, Foiling,edition])

# Write to CSV file
with open(csv_file_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["unique_id", "id", "image_url"])  # Header row
    writer.writerows(cards_data)

print(f"CSV file saved as {csv_file_path}")
