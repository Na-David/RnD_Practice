# 🚨 Don't change the code below 👇
print("Welcome to the Love Calculator!")
name1 = input("What is your name? \n")
name2 = input("What is their name? \n")
# 🚨 Don't change the code above 👆

#Write your code below this line 👇

TRUE = 0

name = name1 + name2
name = name.lower()
t = name.count("t")
r = name.count("r")
u = name.count("u")
e = name.count("e")

true = t+r+u+e

l = name.count("l")
o = name.count("o")
v = name.count("v")
e = name.count("e")

love = l+o+v+e

love_score = str(true)+ str(love)
print(love_score)
loveScore = int(love_score)

if loveScore < 10 and loveScore > 90:
    print(f"Your love score is {loveScore}, you two are like coke and mentos")
elif loveScore >= 40 and loveScore <= 50:
    print(f"Your score is {loveScore}, you are alright together")
else:
    print(f"Your score is {loveScore}")