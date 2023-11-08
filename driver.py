import polars as pl
from pprint import pprint
from equally_weighted_portfolio import EquallyWeightedPortfolio

# Useful when printing
pl.Config.set_tbl_cols(15)
pl.Config.set_tbl_rows(30)

# Read data
data = pl.read_csv('data.csv', try_parse_dates = True)
data2 = pl.read_csv('appl.csv', try_parse_dates = True)

st_vol = (44, 0.7)
lt_vol = (2560, 0.3)

# See class definition for all the parameters
model = EquallyWeightedPortfolio(st_vol, lt_vol, first_ewma=4, n_ewma=4)

# Select you data and run
# See function definition for all the possible params
res = model.run(data2)

print(res.head(10))
print(res.tail(10))

# Save res if you want
# res.write_csv('Results.csv')

percentages = model.compute_percentages()
pprint(percentages)