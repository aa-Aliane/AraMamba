from datasets import load_dataset

ds = load_dataset("asas-ai/ANERCorp")
print(ds["train"].column_names)
print(ds["train"][:15])
