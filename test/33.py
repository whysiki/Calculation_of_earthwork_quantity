import random


# Function to divide a number into n parts
def divide_number(number, parts):
    # Generate n-1 random numbers
    random_numbers = sorted([random.uniform(0, number) for _ in range(parts - 1)])

    # Calculate the differences between consecutive random numbers
    differences = (
        [random_numbers[0]]
        + [random_numbers[i] - random_numbers[i - 1] for i in range(1, parts - 1)]
        + [number - random_numbers[-1]]
    )

    # Sort the differences to make sure they are in ascending order
    differences.sort()

    # Calculate the parts by taking differences between consecutive random numbers
    result = [round(differences[i + 1] - differences[i], 2) for i in range(parts - 1)]

    # Add the last part
    result.append(round(number - differences[-1], 2))

    return result


5
# Number to be divided
number_to_divide = 115.28

# Number of parts
num_parts = 3

# Divide the number into random parts
random_parts = divide_number(number_to_divide, num_parts)

print("Randomly divided parts:", random_parts)
