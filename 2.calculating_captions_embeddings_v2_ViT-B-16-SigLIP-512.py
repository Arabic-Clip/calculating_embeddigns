#%%
import torch
from PIL import Image
import open_clip

import open_clip
import requests
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"You are using {device}")
#%%

import pandas as pd

# wget https://huggingface.co/datasets/Arabic-Clip/ccs_synthetic_translated_arabic_processed/blob/main/processed_dataset_0_12241239_v2.csv

dataset_csv = pd.read_csv("processed_dataset_0_12241239_v2.csv")

#%%
# dataset_csv_5000= dataset_csv[:5000]

dataset_csv_3M_5M = dataset_csv[3000000:5000000]

#%%
dataset_csv_3M_5M
#%%
# show the dataset 
# 12241239 rows × 4 columns
# 200000 rows × 4 columns
# dataset_csv = dataset_csv_5000

#%%

# for idx,row in dataset_3M.iterrows():
#     print(row)
#     print(type(row))
#     print(type(row.to_json()))
#     print(row.to_json(force_ascii=False))
#     break
#%%

import json

from tqdm import tqdm 

jsonl_file_path = 'processed_dataset_csv_3M_5M_siglib_v2.jsonl'

tot = len(dataset_csv_3M_5M)

# Convert the DataFrame to JSONL
with open(jsonl_file_path, 'w') as jsonl_file:
    for _, row in tqdm(dataset_csv_3M_5M.iterrows(), total=tot):
        # Convert each row to a JSON object and write it to the JSONL file
        row_json = row.to_json(force_ascii=False)
        jsonl_file.write(row_json + '\n')

#%%
import json

# Initialize an empty list to store the parsed JSON objects
json_objects = []

# Open the JSONL file for reading
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        # Parse each line as a JSON object
        data = json.loads(line)
        json_objects.append(data)

#%%
totoal_tqdm = len(json_objects)
totoal_tqdm 

#%%
print(json_objects[0])
#%%
#%%
# Iterate over the data and extract the English captions and save them to list

from tqdm.notebook import tqdm

en_captions = []
idx_list = []

for idx_, list_captions in tqdm(enumerate(json_objects), total=totoal_tqdm):

    # print(idx_)
    # print(list_captions)
    en_captions.append(list_captions['caption_en'])
    idx_list.append(list_captions['index'])

#%%
len(en_captions), len(idx_list)
#%%
en_captions
#%%
idx_list
#%%
#%%
# import torch
# from PIL import Image
# import open_clip


# device = "cuda" if torch.cuda.is_available() else "cpu"

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
# tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')
# model.to(device)

import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

device = "cuda" if torch.cuda.is_available() else "cpu"


# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
# tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')


model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-512')
tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-512')

#%%
# Check the device of the model's parameters

#%%
# Check the device of the model's parameters
model.to('cuda')

# Check the device of the model's parameters
device = next(model.parameters()).device

if device.type == 'cuda':
    print("Model is already on GPU")
else:
    # Move the model to GPU
    model = model.to('cuda')
    print("Model moved to GPU")

#%%


#%%

class SimpleTextDataset(torch.utils.data.Dataset):
    """ Extend the ability of the dataset loader """
    def __init__(self, texts):
        'Initialization'
        self.texts = texts
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.texts)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        return self.texts[idx]

text_dataset = SimpleTextDataset(en_captions)
#%%

len(text_dataset)

#%%
text_loader = torch.utils.data.DataLoader(
    text_dataset, 
    batch_size = 512,
    shuffle=False
)
#%%
len(text_loader)

#%%

len(en_captions) / 512
#%%

# Check the output for one batch => For Debugging purposes only 

# iterator = iter(text_loader)
# text_sample = next(iterator)

# encoded_input = tokenizer(text_sample)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# encoded_input = encoded_input.to(device)

# with torch.no_grad(), torch.cuda.amp.autocast():
#     text_features = model.encode_text(encoded_input)

#     print(type(text_features))
#     print(text_features.shape)
#     print(text_features)
#%%

output_features = list()

import sys
for i, texts in tqdm(enumerate(text_loader), total=len(text_loader)):

    # print("Current batch is {}/{}".format(i,len(text_loader)))

    # print(device)

    # print(texts)
    # print(len(texts))

    # Tokenize sentences
    encoded_input = tokenizer(texts)
    

    # Move the encoded input to the GPU
    encoded_input = encoded_input.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(encoded_input)
        output_features.append(text_features)
    
        # print(type(text_features))
        # print(text_features.shape)
        # print(text_features)
#%%
len(output_features[0])

#%%
len(output_features)

# %%
output_features[0].shape # torch.Size([100, 640])

#%%
output_features[0][0]

#%%
output_features[0][1]
#%%
#%%
# import pickle

# # File path where you want to save the pickle file
# file_path = "output_features_0_5000.pkl"

# # Open the file in binary write mode and save the list
# with open(file_path, 'wb') as file:
#     pickle.dump(output_features, file)

# #%%
# # I am here 
# import pickle

# file_path = "output_features_0_5000.pkl"

# with open(file_path, 'rb') as file:
#     output_features = pickle.load(file)

# print(f"List saved as a pickle file at {file_path}")

#%%

# Create a json of ['id', 'embedding'] for each caption

en_captions_json = {}

# idx_list_supp = range(0,len(idx_list))

# len(idx_list_supp)
#%%
len(idx_list)
# idx_list[100-1]
len(output_features)
#%%
from tqdm import tqdm 

idx_sp = 0

tot = len(output_features)

print(tot)

for item in tqdm(output_features, total=tot):
    # print(f"length of item {len(item)}")
    # print(item)

    for item2 in item:

        en_captions_json[idx_list[idx_sp]] = item2.tolist()

        idx_sp += 1
        # print(len(tensor_list))
#%%
idx_list[0]
# %%

for i,k in en_captions_json.items():
    print(i, ' ',k)
    break
# %%
total_en_captions_json = len(en_captions_json.keys())
# # %%
# for i, j in en_captions_json.items():
#     print(i, j)
#     break
total_en_captions_json
#%%
# total_en_captions_json
#%%

# # Save the json file
# import json

# with open("en_captions_json.json", "w") as outfile: 
#     json.dump(en_captions_json, outfile, indent=4)

#%%


#%%
json_objects[0]
#%%
# output_json

# Add all other information to the json file and save it to create a full dataset:

import json
from tqdm.notebook import tqdm

with open("processed_dataset_csv_3M_5M_siglib_v2.jsonl",'w') as file:
    
    for idx_, data_captions in tqdm(enumerate(en_captions_json.items()), total=total_en_captions_json):

        # print(idx_)
        key_dict, val_dict = data_captions
        # print(key_dict)
        # print(val_dict)
        # print(json_objects[idx_])

        json_object = {
            "index": key_dict,
            "url":  json_objects[idx_]["url"],
            "caption_en": json_objects[idx_]["caption_en"],
            "embeddings_en": [val_dict],
            "caption_ar": json_objects[idx_]["caption_ar"],            
            }
        # print(json_object)
        # break
        json.dump(json_object, file,ensure_ascii=False)
        file.write("\n")

#%%
key_dict
#%%

json_objects[0]

#%%

with open("processed_dataset_csv_3M_5M_siglib_v2.jsonl",'r') as file:
    
    print(type(file))

    count_captions = [1 for i in file]
    print(f"Full count of captions: {sum(count_captions)}")

    # for itm in file:
    #     print(json.loads(itm))
    #     break

#%%
print(f"Full count of captions: {sum(count_captions)}")

#%%


## Confirm the generated embeddings manually

import torch
from PIL import Image
import requests
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-512')
tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-512')

model.to(device)

text = "person pointing at a wall mounted magnetic chalk board"

# Tokenize sentences
encoded_input = tokenizer(text)


# Move the encoded input to the GPU
encoded_input = encoded_input.to(device)


with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(encoded_input)
    print(text_features.tolist())
# %%