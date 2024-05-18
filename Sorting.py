import random
import string


import random
# depending on the function it will sort the item depending on the first item in the set 


#sort by price?


#sort by number?
# sort by rarity 
#sort by set#=value will return in 000/xxx-price in tcg player price in usd


def generate_formatted_strings(count, identifier):
    result = []
    for _ in range(count):
        # Use the specified identifier "OUT"
        part1 = identifier
        # Generate the second part with three digits
        part2 = ''.join(random.choice('0123456789') for _ in range(3))
        # Generate the price part after the dash with three digits before the dot and two after
        price_part = f"{random.randint(0, 999):03}.{random.randint(0, 99):02}"
        
        # Combine all parts into the final format
        formatted_string = f"{part1}{part2}-{price_part}"
        result.append(formatted_string)
    
    return result

# Example usage with the identifier "OUT"
formatted_strings = generate_formatted_strings(5, "OUT")
print(formatted_strings)



# Example usage




def creatematrix(row,colum,items):
    matrix=[]
    value=0
    for r in range(row):
        row=[]
        for c in range(colum):
            row.append(0)
            value_index += 1
        matrix.append(row)
    return matrix

def print_matrix(matrix):
 
    for row in matrix:
        print(' '.join(map(str, row)))

if __name__ == "__main__":
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    matrix = creatematrix(rows, cols)
    print("Generated matrix:")
    #print_matrix(matrix)
        
    
