import pandas as pd

df = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")



# print(df.columns)
# Country 
# Year
# Attack Type
# Target Industry
# Financial Loss (in Million $)
# Number of Affected Users
# Attack Source
# Security Vulnerability Type
# Defense Mechanism Used
# Incident Resolution Time (in Hours)

# print(df["Attack Type"].value_counts())
# DDoS                 531
# Phishing             529
# SQL Injection        503
# Ransomware           493
# Malware              485
# Man-in-the-Middle    459

# print(df["Country"].value_counts())
# UK           321
# Brazil       310
# India        308
# Japan        305
# France       305
# Australia    297
# Russia       295
# Germany      291
# USA          287
# China        281

#print(df.isna().value_counts())
#3000 false, so should be good

#UK has the highest number of attacks, what are they? - phishing
def uk_attacks():
    uknum = df[df["Country"] == "UK"]
    # uknum = uknum.groupby("Attack Type")["Defense Mechanism Used"].agg(lambda x: x.mode().iloc[0])
    print("Most used attack: ")
    attacks = uknum["Attack Type"].value_counts()
    print(attacks)
    print("\n")

    print("Most used Defense: ")
    defense = uknum["Defense Mechanism Used"].value_counts()
    print(defense)
    print("\n")
    
    # df["Attack Type"].corr(df["Defense Mechanism Used"])
    print("Most used Defense per Attack: ")
    uk = uknum.groupby("Attack Type")["Defense Mechanism Used"].value_counts()
    print(uk)
    print("\n")

    print("Most used Attack per Defense: ")
    uk = uknum.groupby("Defense Mechanism Used")["Attack Type"].value_counts()
    print(uk)
    print("\n")
uk_attacks()

# which country has the biggest loss? - UK
def fin_loss():
    fin = df.groupby("Country").agg({"Financial Loss (in Million $)": "sum"})
    print(fin.sort_values(by="Financial Loss (in Million $)", ascending=False))
# fin_loss()

# What makes the most amount of damage?
def Atk_damage():
    fin = df.groupby("Attack Type").agg({"Financial Loss (in Million $)": "sum"})
    print(fin.sort_values(by="Financial Loss (in Million $)", ascending=False))
# Atk_damage()

# what is the avg time to fix attack?
def Atk_time():
    fin = df.groupby("Attack Type").agg({"Incident Resolution Time (in Hours)": "mean"})
    print(fin.sort_values(by="Incident Resolution Time (in Hours)", ascending=False))
# Atk_time()


# is there an attack that some more the most?
def atk_overall():
    over = df.groupby("Attack Type")["Country"].value_counts()
    print(over)
atk_overall()