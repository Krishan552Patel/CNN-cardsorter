import random

def generate_formatted_strings(count, identifier):
    result = []
    for _ in range(count):
        part1 = identifier
        part2 = ''.join(random.choice('0123456789') for _ in range(3))
        price_part = f"{random.randint(0, 999):03}.{random.randint(0, 99):02}"
        random_char = random.choice(['t', 'c', 'r', 'm', 'L', 'f'])
        formatted_string = f"{part1}{part2}-{price_part}/{random_char}/"
        result.append(formatted_string)
    
    return result

def sort_card_into_bin(card, bins, misc_bin, max_capacity=32, max_sub_bins=3):
    # Extract the letter from the card
    letter = card.split('/')[1]  # Extract the letter
    if letter not in bins:
        bins[letter] = [[]]  # Initialize a list with one bin if it doesn't exist
    
    # Check if the number of sub-bins has reached the limit
    if len(bins[letter]) >= max_sub_bins and len(bins[letter][-1]) >= max_capacity:
        misc_bin.append(card)
        return f"Card '{card}' placed in miscellaneous bin"
    
    # Check if the last bin for this letter has reached max capacity
    if len(bins[letter][-1]) >= max_capacity:
        bins[letter].append([])  # Add a new bin
    
    current_bin_index = len(bins[letter]) - 1
    bins[letter][current_bin_index].append(card)
    return f"Card '{card}' placed in bin '{letter}-{current_bin_index}'"

# Initialize the bins
bins = {}
misc_bin = []

# Generate formatted strings
formatted_strings = generate_formatted_strings(1000, "OUT")
print("Generated card strings:", formatted_strings)

# Process each card one at a time
for card in formatted_strings:
    result = sort_card_into_bin(card, bins, misc_bin)
    print(result)

# Print final bins to verify
print("Final bins:")
for letter, bin_lists in bins.items():
    for idx, bin_contents in enumerate(bin_lists):
        print(f"Bin '{letter}-{idx}': {bin_contents}")
print(f"Miscellaneous bin: {misc_bin}")
