import torch
from tqdm import tqdm


# Verify that authors, venues, and papers in a previous yaer still appear in the next year in the data as first few elements
def verify_yearly_coherence():
    past_data = torch.load('data/yearly_snapshots_specter2/G_2005.pt')
    for year in tqdm(range(2006, 2025)):
        data = torch.load(f'data/yearly_snapshots_specter2/G_{year}.pt')
        for data_type in ['author', 'venue', 'paper']:
            previous_ids = past_data[data_type]['raw_ids']
            current_ids = data[data_type]['raw_ids']
            assert previous_ids == current_ids[:len(previous_ids)]
        past_data = data


verify_yearly_coherence()
