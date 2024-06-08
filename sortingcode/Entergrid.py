def create_grid(length, width):
    # Initialize an empty grid
    grid = []

    # Fill the grid with empty bins
    for i in range(length):
        row = []
        for j in range(width):
            row.append('Bin')  # You can customize what you want to store in each bin
        grid.append(row)

    return grid

def print_grid(grid):
    for row in grid:
        print(' | '.join(row))
    print()

def main():
    # Get user input for the grid dimensions
    try:
        length = int(input("Enter the length of the grid: "))
        width = int(input("Enter the width of the grid: "))
    except ValueError:
        print("Please enter valid integer values for length and width.")
        return

    # Create the grid
    grid = create_grid(length, width)

    # Print the created grid
    print("Created Grid:")
    print_grid(grid)

if __name__ == "__main__":
    main()