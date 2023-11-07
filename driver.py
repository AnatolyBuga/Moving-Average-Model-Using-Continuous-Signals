import polars as pl
from pprint import pprint
from equally_weighted_portfolio import EquallyWeightedPortfolio

pl.Config.set_tbl_cols(15)
pl.Config.set_tbl_rows(30)

data = pl.read_csv('data.csv', try_parse_dates = True)
data2 = pl.read_csv('appl.csv', try_parse_dates = True)

st_vol = (44, 0.7)
lt_vol = (2560, 0.3)

model = EquallyWeightedPortfolio(st_vol, lt_vol, first_ewma=4, n_ewma=4)
res = model.run(data2)
percentages = model.compute_percentages()
pprint(percentages)