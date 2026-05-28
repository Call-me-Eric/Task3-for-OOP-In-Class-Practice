from datasets import load_dataset

ds = load_dataset("ylecun/mnist")

with open('output.txt', 'w', encoding='utf-8') as file:
    print(ds,file = file)