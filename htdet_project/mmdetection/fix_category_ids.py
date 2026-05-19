import json

files = [
    "data/urpc/annotations/instances_train2018.json",
    "data/urpc/annotations/instances_val2018.json"
]

for file in files:
    with open(file) as f:
        data = json.load(f)

    # shift annotation category ids
    for ann in data["annotations"]:
        ann["category_id"] += 1

    # shift category ids
    for cat in data["categories"]:
        cat["id"] += 1

    with open(file, "w") as f:
        json.dump(data, f)

print("Category IDs shifted successfully")