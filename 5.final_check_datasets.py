#%%
from datasets import load_dataset

ds_train = load_dataset("Arabic-Clip/Arabic_3M_5M_ViT-B-16-SigLIP-512", split="train")

ds_train

# Dataset({
#     features: ['index', 'url', 'en_caption', 'embeddings_en', 'caption_ar'],
#     num_rows: 2000000
# })
#%%
ds_validation = load_dataset("Arabic-Clip/Arabic_3M_5M_ViT-B-16-SigLIP-512", split="validation")

ds_validation

# Dataset({
#     features: ['index', 'url', 'en_caption', 'embeddings_en', 'caption_ar'],
#     num_rows: 5000
# })
# %%
#%%
from datasets import load_dataset

ds_train = load_dataset("Arabic-Clip/Arabic_3M_5M_ViT-B-16-plus-240", split="train")

ds_train

# Dataset({
#     features: ['index', 'url', 'en_caption', 'embeddings_en', 'caption_ar'],
#     num_rows: 200000
# })
#%%
ds_validation = load_dataset("Arabic-Clip/Arabic_3M_5M_ViT-B-16-plus-240", split="validation")

ds_validation

# Dataset({
#     features: ['index', 'url', 'en_caption', 'embeddings_en', 'caption_ar'],
#     num_rows: 5000
# })
# %%

from datasets import load_dataset

ds_train = load_dataset("Arabic-Clip/Arabic_MSCOCO_1st_ViT-B-16-plus-240")
ds_train

# DatasetDict({
#     train: Dataset({
#         features: ['id', 'caption_en', 'embeddings_en', 'caption_ar'],
#         num_rows: 113281
#     })
# })
# %%
from datasets import load_dataset

ds_train = load_dataset("Arabic-Clip/Arabic_MSCOCO_1st_ViT-B-16-SigLIP-512")
ds_train

# DatasetDict({
#     train: Dataset({
#         features: ['id', 'caption_en', 'embeddings_en', 'caption_ar'],
#         num_rows: 11454
#     })
# })
