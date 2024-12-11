# Stats pull from the data sets we used for our project

import pandas

from src.dataset._dataset_utils import create_datasets

symbols = [
  "atnf",
  "bivi",
  "cycc",
  "vtak",
  "spx",
]

datasets = {sym: create_datasets(sym) for sym in symbols}

results = {
  'Symbol': [],
  'Train Sell': [],
  'Train Hold': [],
  'Train Buy': [],
  'Valid Sell': [],
  'Valid Hold': [],
  'Valid Buy': [],
  'Test Sell': [],
  'Test Hold': [],
  'Test Buy': [],
  'Total Sell': [],
  'Total Hold': [],
  'Total Buy': [],
}
for sym in datasets:
  train, valid, test = datasets[sym]

  results['Symbol'].append(sym)
  results['Train Sell'].append(train.target_counts[0].item())
  results['Train Hold'].append(train.target_counts[1].item())
  results['Train Buy'].append(train.target_counts[2].item())
  results['Valid Sell'].append(valid.target_counts[0].item())
  results['Valid Hold'].append(valid.target_counts[1].item())
  results['Valid Buy'].append(valid.target_counts[2].item())
  results['Test Sell'].append(test.target_counts[0].item())
  results['Test Hold'].append(test.target_counts[1].item())
  results['Test Buy'].append(test.target_counts[2].item())
  results['Total Sell'].append((train.target_counts[0] + valid.target_counts[0] + test.target_counts[0]).item())
  results['Total Hold'].append((train.target_counts[1] + valid.target_counts[1] + test.target_counts[1]).item())
  results['Total Buy'].append((train.target_counts[2] + valid.target_counts[2] + test.target_counts[2]).item())

df = pandas.DataFrame(results)
df.to_csv('./data/category_counts_by_symbol.csv', index=False)