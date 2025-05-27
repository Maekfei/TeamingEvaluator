import os

# Set the TOKENIZERS_PARALLELISM environment variable to false to avoid warnings and potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set visible GPUs before importing PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_aug2023refresh_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_aug2023refresh_base')

# Load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2_aug2023refresh", source="hf", load_as="proximity", set_active=True)

# Prepare new documents and IDs
new_docs = []
new_ids = []
for index, row in filtered_df.iterrows():
    title = row['Title']
    abstract = row['Abstract']
    pmid = row['PMID']
    new_docs.append(str(title) + tokenizer.sep_token + str(abstract))
    new_ids.append(pmid)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 100  # Define an appropriate batch size
all_embeddings = []
all_ids = []

class DocumentDataset(Dataset):
    def __init__(self, docs, ids):
        self.docs = docs
        self.ids = ids

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx], self.ids[idx]

dataset = DocumentDataset(new_docs, new_ids)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Function to process a single batch
def process_batch(batch):
    docs, ids = batch
    inputs = tokenizer(list(docs), padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    inputs = inputs.to(device)  # Move inputs to GPU
    with torch.no_grad():  # Disable gradient calculations
        output = model(**inputs)
    batch_embeddings = output.last_hidden_state[:, 0, :].cpu()  # Move embeddings back to CPU
    return batch_embeddings, ids

# Process the dataset in batches
for batch in tqdm(dataloader):
    batch_embeddings, batch_ids = process_batch(batch)
    
    all_embeddings.append(batch_embeddings)
    all_ids.extend(batch_ids)

new_embeddings = torch.cat(all_embeddings, dim=0).numpy()
new_ids = all_ids

print("Embeddings have been generated")