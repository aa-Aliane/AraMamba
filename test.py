from datasets import load_dataset

ds = load_dataset("Elnagara/hard")
print(ds["train"].features)
print(ds["train"][0])
