import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

data_dir = Path(__file__).parent # we can change this later if we want to move the data into its own folder
documents = []
ratings = []

for file_path in sorted(data_dir.glob("Health_and_Personal_Care.part-*.jsonl")):
    with file_path.open(encoding="utf-8") as file:
        for line in file:
            review = json.loads(line)
            rating = int(float(review["rating"]))
            if rating == 3:
                continue

            title = review.get("title") or ""
            text = review.get("text") or ""
            document = f"{title.strip()} {text.strip()}".strip() # combining title and text for td-idf vectorization

            if document:
                documents.append(document) # combined txt
                ratings.append(rating) # rating is rating
