import sys

def add_numbers(num1, num2):
    return num1 + num2

if __name__ == "__main__":
    # Read the command-line arguments
    num1 = float(sys.argv[1])
    num2 = float(sys.argv[2])

    # Perform addition and print the result
    result = add_numbers(num1, num2)
    print(result)
