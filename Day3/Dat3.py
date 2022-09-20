print("Welcome to the rollercoaster!")
height = int(input("What is your height in cm? "))
bill = 0

if height >= 120:
    age = int(input("What is your age?"))
    if age <= 12:
        bill = 5
        print(f"Please pay ${bill}.")
    elif age <= 18:
        bill = 6
        print(f"Please pay ${bill}.")
    elif age >= 45 and age <= 55:
        bill = 0
        print("Everything is going to be OK, Have a free ride on us!")
    else:
        bill = 8
        print(f"Please pay ${bill}")

    photo = input("Do you want a photo taken? Y or N.")
    if photo == "Y":
        bill += 3
        print(f"Your final bill is ${bill}")

else:
    print("Sorry, you have to grow taller before you can ride.")
