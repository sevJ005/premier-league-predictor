import csv

data = [
    "pl_2018-2019.csv", "pl_2019-2020.csv", "pl_2020-2021.csv",
    "pl_2021-2022.csv", "pl_2022-2023.csv",
    "pl_2023-2024.csv", "pl_2024-2025.csv", "pl_2025-2026.csv"
]
unique = set()

for file in data:
    with open(f"data/raw/{file}", mode="r", newline="", encoding="utf-8") as doc:
        reader = csv.DictReader(doc)
        for row in reader:
            unique.add(row["HomeTeam"])
            unique.add(row["AwayTeam"])

print(unique)
print(len(unique))
