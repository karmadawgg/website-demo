
# numberCheck.py

def is_prime(n):
    # Numbers less than 2 are not prime
    if n < 2:
        return False

    # Check divisibility
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True


def is_palindrome(n):
    # Convert number to string
    num = str(n)

    # Check whether it reads the same backward
    return num == num[::-1]
