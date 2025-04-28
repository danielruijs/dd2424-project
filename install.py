import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("shubhammaindola/harry-potter-books")

print("Harry Potter books downloaded to: ", path)

# List all text files
file_paths = [
    os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(".txt")
]

# Read all text files and concatenate them
texts = []
for path in file_paths:
    with open(path, "r", encoding="utf-8") as f:
        texts.append(f.read())

# Concatenate all texts into a single string
text = "\n".join(texts)

print("Harry Potter books loaded. Total characters: ", len(text))
print(text[:1000])  # Print the first 1000 characters of the text
