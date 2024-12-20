# project-proposal

# Data Dictionary

| Col | Name      | Description                         |
|-----|-----------|-------------------------------------|
| 0   | timestamp | UNIX timestamp e.g. - 1538031600    |
| 1   | open      | Opening price of the stock          |
| 2   | high      | Highest price of the stock          |
| 3   | low       | Lowest price of the stock           |
| 4   | close     | Closing price of the stock          |
| 5   | adj_close | Adjusted closing price of the stock |
| 6   | volume    | Volume of the stock traded          |
| 7   | year      | Year                                |
| 8   | month     | Month                               |
| 9   | day       | Day                                 |

# for further information: installing pytorch in conda with cuda

1. create env `conda create -n dl-group-project -y`
2. install pytorch, cuda, etc: `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y`
3. install remaining requirements: `pip install -r requirements.txt`
