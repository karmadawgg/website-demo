
# Import the module
import numberCheck

# Take integer input
num = int(input("Enter an integer: "))

# Check whether the number is prime
if numberCheck.is_prime(num):
    print(num, "is a Prime Number")
else:
    print(num, "is not a Prime Number")

# Check whether the number is palindrome
if numberCheck.is_palindrome(num):
    print(num, "is a Palindrome Number")
else:
    print(num, "is not a Palindrome Number")
