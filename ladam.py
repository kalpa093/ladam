import torch
import random
from transformers import BertTokenizer, BertModel
import csv
import numpy as np
import re
import argparse
import os


def clear_line(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


def calculate_attention_score(sentence1, sentence2, tokenizer, model):

    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding=True)
    inputs.to(device)
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    attention_scores = attentions[0][0]
    attention_scores = attention_scores.detach().cpu().numpy()
    num_tokens1 = len(tokenizer.tokenize(sentence1)) + 2  # Including [CLS] and [SEP]
    num_tokens2 = len(tokenizer.tokenize(sentence2)) + 1  # Including only [SEP]

    cross_attention_scores = attention_scores[:, :num_tokens1, num_tokens1:num_tokens1 + num_tokens2].mean(axis=0)

    random_word_index = np.random.choice(len(sentence1.split())) + 1
    selected_word = np.argmax(cross_attention_scores[random_word_index])
    sentence1_tokens = tokenizer.tokenize(sentence1)
    sentence2_tokens = tokenizer.tokenize(sentence2)
    sentence1_new = sentence1.replace(sentence1_tokens[random_word_index - 1], sentence2_tokens[selected_word])

    return sentence1_new


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset file name (e.g., test.csv)")
parser.add_argument("-n", "--n_aug", type=int, required=True, help="Number of augmentation (e.g., 1, 2, ... ", default=2)
args = parser.parse_args()

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)

label = []
data = []
new_data = []
dataset_name = re.split(r'[_.]', args.dataset)[0]
aug_n = args.n_aug

with open(f"dataset/{dataset_name}.csv", 'r', encoding='Windows-1252') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append(clear_line(row[0]))
        label.append(row[1])


for j in range(aug_n):
    for i in range(len(data)):
        try:
            if len(data[i]) > 1:
                random_index = random.randint(0, len(data) - 1)
                while random_index == i or label[i] != label[random_index]:
                    random_index = random.randint(0, len(data) - 1)
                sentence1 = data[i]
                sentence2 = data[random_index]
                new_sen = calculate_attention_score(sentence1, sentence2, tokenizer, model)
                new_data.append([new_sen, label[i]])
        except Exception as E:
            pass


with open(f"dataset/{dataset_name}_ladam_{aug_n}.csv", "w", newline='', encoding='Windows-1252') as f:
    write = csv.writer(f)
    write.writerow(['sentence', 'label'])
    for i in range(len(data)):
        if len(data[i]) > 1:
            write.writerow([data[i], label[i]])
    for entry in new_data:
        write.writerow(entry)


