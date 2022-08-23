from utils import *
btc = Extract().format_df()
model = Model(btc[0], 1, 3, btc[1])

model.display(save_fig=True)
model.display(save_fig=True, ax_log=True)
model.save_as_csv()
