"""Generates a small SAMPLE dataset (cyberbullying_sample.csv) so the app runs out of the box.

For real results, replace the CSV with the Kaggle "Cyberbullying Classification" dataset
(columns tweet_text, cyberbullying_type) - app.py maps it automatically.
"""
import random
import pandas as pd

random.seed(42)

targets = ["you", "you", "you", "u", "this guy", "she", "he", "that girl", "that kid", "your whole family", "the new girl"]
insults = ["stupid", "ugly", "worthless", "pathetic", "a loser", "a freak", "disgusting", "useless", "an idiot", "a total joke",
           "fat and ugly", "so dumb", "a waste of space", "trash", "a clown"]
bully_templates = [
    "{t} are {i} and nobody likes {t2}",
    "you are {i}, everyone is laughing at you",
    "why are you even alive, you are {i}",
    "go kill yourself you {i1}",
    "nobody wants you here, just leave",
    "i will find you and make you regret this",
    "{t} is {i}, what a loser lol",
    "shut up you {i1}, no one asked for your opinion",
    "everyone hates you, you {i1}",
    "look at this {i1} trying to be cool, so cringe",
    "you will never be good enough, give up already",
    "i hope something bad happens to you, {i1}",
    "delete your account you {i1}, nobody cares about you",
    "you are so {i1} it hurts to look at you",
    "we all talk about how {i} you are behind your back",
    "keep posting and we will make your life hell",
    "{t} deserve to be alone because you are {i}",
    "what a pathetic {i1}, do the world a favor and disappear",
    "she is {i}, don't talk to her or sit with us",
    "you ugly {i1}, nobody will ever love you",
    "i know where you live, watch your back",
    "stop being such a {i1} and get lost",
    "everyone in class thinks you are {i}",
    "no wonder you have no friends, you are {i}",
]
good_templates = [
    "had a great day at the park with friends today",
    "congratulations on your new job, you deserve it",
    "thanks for helping me with the assignment, really appreciate it",
    "the match last night was amazing, what a finish",
    "i love this new song, been playing it all day",
    "can anyone recommend a good book on machine learning",
    "happy birthday, hope you have a wonderful day",
    "the weather is lovely this evening, perfect for a walk",
    "great presentation today, you explained everything clearly",
    "i disagree with this policy but i respect your opinion",
    "the central bank announced its interest rate decision on wednesday",
    "just finished my project report, feeling relieved",
    "this recipe turned out really well, thanks for sharing",
    "looking forward to the weekend trip with family",
    "i really hate monday mornings, so tired",
    "this exam is killing me, need more coffee",
    "that movie was so bad it was funny, we laughed the whole time",
    "i am so stupid, forgot my umbrella again",
    "the traffic today was terrible, took two hours to get home",
    "proud of you for finishing the marathon",
    "does anyone know when the college fest registrations open",
    "nice work on the design, the colours look great",
    "we lost the game but the team played really well",
    "my phone battery is useless these days, need a new one",
    "thank you everyone for the support, it means a lot",
    "let us meet at the library after class to study together",
    "the new update is slow and buggy, hope they fix it soon",
    "i am so excited for the concert next week",
    "sorry i was late, the train got delayed",
    "you are a really good friend, thanks for always being there",
    "the food at that restaurant was disappointing, would not go again",
    "please share the notes from yesterday's lecture",
    "what a beautiful sunset, look at those colours",
    "good luck for your interview tomorrow, you will do great",
    "i am not a fan of this show but my friends love it",
]
fillers = ["", "", "", " lol", " omg", " tbh", " #news", " @user", " http://t.co/xyz", "!!", " ...", " smh"]
typos = [("you", "u"), ("are", "r"), ("because", "bcoz"), ("people", "ppl"), ("really", "rly")]


def noise(s):
    if random.random() < 0.25:
        a, b = random.choice(typos)
        s = s.replace(a, b, 1)
    s = s + random.choice(fillers)
    if random.random() < 0.15:
        s = s.upper()
    elif random.random() < 0.3:
        s = s.capitalize()
    return s


rows = []
for _ in range(700):
    t = random.choice(bully_templates)
    s = t.format(t=random.choice(targets), t2=random.choice(["you", "u", "anyone", "them"]),
                 i=random.choice(insults), i1=random.choice(["idiot", "loser", "freak", "moron", "clown", "coward", "pathetic loser"]))
    rows.append((noise(s), "Bullying"))
for _ in range(700):
    s = random.choice(good_templates)
    rows.append((noise(s), "Not Bullying"))

df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates().sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv("cyberbullying_sample.csv", index=False)
print(df.label.value_counts())
