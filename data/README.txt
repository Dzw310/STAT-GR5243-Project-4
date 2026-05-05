DATA SOURCE
===========

The raw CSV files are too large to include in this repository (~3.5 GB total).

Download the LendingClub Loan Data (2007-2018 Q4) from Kaggle:

  https://www.kaggle.com/datasets/wordsforthewise/lending-club

After downloading, place the following files in this directory:

  data/
  ├── accepted_2007_to_2018Q4.csv   (~1.7 GB, 2,260,701 loans)
  └── rejected_2007_to_2018Q4.csv   (~1.8 GB, 27.6M rejected applications)

Then run the notebooks in order (01 -> 02 -> 03 -> 04).
Intermediate parquet files (cleaned.parquet, engineered.parquet, etc.)
will be generated automatically by the notebooks.
