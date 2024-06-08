
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
def sort_card_into_bin(card, bins, misc_bin, max_capacity=32, max_sub_bins=3):## Need to add a sorting by identifier 
    # Extract the letter from the card
    id = card[1]  # Extract the letter
    if id  not in bins:
        bins[id] = [[]]  # Initialize a list with one bin if it doesn't exist
    
    # Check if the number of sub-bins has reached the limit
    if len(bins[id]) >= max_sub_bins and len(bins[id][-1]) >= max_capacity:
        misc_bin.append(card)
        return f"Card '{card}' placed in miscellaneous bin"
    
    # Check if the last bin for this letter has reached max capacity
    if len(bins[id][-1]) >= max_capacity:
        bins[id].append([])  # Add a new bin
    
    current_bin_index = len(bins[id]) - 1
    bins[id][current_bin_index].append(card)
    return f"Card '{card}' placed in bin '{id}-{current_bin_index}'"
