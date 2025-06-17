import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = '/data/jx4237data/GNNteamingEvaluator/TeamingEvaluator/data/yearly_snapshots_specter2_starting_from_year_1/G_2024.pt'
G = torch.load(file_path)

# Get citation data
citations = G['paper']['y_citations'].numpy()

# Calculate mean and standard error for each year
mean_citations = np.mean(citations, axis=0)
std_citations = np.std(citations, axis=0)
std_error = std_citations / np.sqrt(len(citations))

# Create the plot
years = np.arange(1, 6)  # Years 1-5
plt.figure(figsize=(10, 6))
plt.errorbar(years, mean_citations, yerr=std_error, fmt='o-', capsize=5, capthick=1, ecolor='gray', color='blue')

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Average Citations')
plt.title('Average Citations per Year with Standard Error')
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('citation_trend.png', dpi=300, bbox_inches='tight')
plt.close() 