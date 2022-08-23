import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit
from uncertainties import ufloat
from datetime import datetime

pd.plotting.register_matplotlib_converters()
sns.set_theme(style="whitegrid", palette="rainbow")


# This class extracts ticker info and formats df
class Extract:
    def __init__(self):
        self.ticker = input('Ticker: ')
        self.start = input('Start Date: ')
        self.end = input('End Date: ')

        if self.ticker == '':
            self.ticker = 'BTC-USD'
        if self.start == '':
            self.start = '01-01-2017'
        if self.end == '':
            self.end = datetime.today().strftime('%Y-%m-%d')

    def format_df(self):
        df = yf.download(self.ticker)
        df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = df[
            ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].apply(pd.to_numeric)
        df['avg'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        df = df.reset_index()
        df['Numeric time (ns)'] = df['Date'].apply(lambda d: d.value)
        df = df.dropna(how='all', subset=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        df = df[(self.start <= df['Date']) & (df['Date'] <= self.end)]
        df['Day Actual'] = df['Numeric time (ns)'] * (10 ** 9 * 60 ** 2 * 24) ** -1
        # Day is really just the index in this case but leave it like this incase
        df['Day'] = (df['Numeric time (ns)'] - df['Numeric time (ns)'].iloc[0]) * (10 ** 9 * 60 ** 2 * 24) ** -1
        df = df.dropna(how='all', subset=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']).reset_index(drop=True)

        return df, self.ticker

    # n is multiple of timeframe
    # s is number is std_dev


class Model:
    def __init__(self, df, n, s, ticker):
        self.df = df
        self.n = n
        self.s = s
        self.ticker = ticker
        self.act_date = self.df['Day Actual']
        self.x_data = self.df['Day']
        self.y_data = self.df[['Open', 'High', 'Low', 'Close']]
        self.fit()

    def fit(self):
        func = lambda x, a, b, c: a * x ** b + c

        y_means = np.array(self.y_data).mean(axis=1)
        y_spread = np.array(self.y_data).std(axis=1)

        best_fit_ab, covar = curve_fit(func, self.x_data, y_means,
                                       sigma=y_spread,
                                       absolute_sigma=False, maxfev=1000)
        sigma_ab = np.sqrt(np.diagonal(covar))
        a = ufloat(best_fit_ab[0], sigma_ab[0])
        b = ufloat(best_fit_ab[1], sigma_ab[1])
        c = ufloat(best_fit_ab[2], sigma_ab[2])
        text_res = "y=a*x^b+c\nBest fit parameters:\na = {}\nb = {}\nc = {}".format(a, b, c)

        hires_x = self.n * self.x_data

        bound_lower = [func(hires_x, *(best_fit_ab - self.s * sigma_ab)) for self.s in range(1, self.s + 1)]
        bound_lower_rolling = [y_means-self.s*pd.Series(y_spread).rolling(100).std() for self.s in range(1, self.s + 1)]
        fit_vals = func(hires_x, *best_fit_ab)
        bound_upper = [func(hires_x, *(best_fit_ab + self.s * sigma_ab)) for self.s in range(1, self.s + 1)]
        bound_upper_rolling = [y_means + self.s * pd.Series(y_spread).rolling(100).std() for self.s in range(1, self.s + 1)]

        x_calc = ((hires_x + self.act_date[0]) * (10 ** 9 * 60 ** 2 * 24)).apply(
            pd.to_datetime)
        y_calc = fit_vals
        y_low = bound_lower
        y_up = bound_upper

        return {'fit_params': text_res,
                'x_for_fit': hires_x,
                'lower_bound': bound_lower,
                'fit_vals': fit_vals,
                'upper_bound': bound_upper,
                'x_shift': x_calc,
                'y_calc': y_calc,
                'y_low': y_low,
                'y_low_roll': bound_lower_rolling,
                'y_up': y_up,
                'y_up_roll': bound_upper_rolling}

    def save_as_csv(self):
        x_shift = self.fit()['x_shift']
        y_calc = self.fit()['y_calc']
        y_low = self.fit()['y_low']
        y_up = self.fit()['y_up']
        df1 = self.df

        df1['y_calc'] = y_calc

        for i in range(0, self.s):
            df1[f'-{i + 1} * low_std_dev'] = y_low[i]
            df1[f'{i + 1} * high_std_dev'] = y_up[i]
            df1[f'{i + 1} * low_std_dev_rolling'] = self.fit()['y_low_roll'][i]
            df1[f'{i + 1} * high_std_dev_rolling'] = self.fit()['y_up_roll'][i]
        result = df1.to_csv('saved_df.csv')
        return result

    def display(self, save_fig=False, ax_log=False):
        x_shift = self.fit()['x_shift']
        y_calc = self.fit()['y_calc']
        y_low, y_low_roll = self.fit()['y_low'], self.fit()['y_low_roll']
        y_up, y_up_roll = self.fit()['y_up'], self.fit()['y_up_roll']
        x_roll = range(int(self.df['Day Actual'].values[0]), int(self.df['Day Actual'].values[0]) + len(y_up_roll[0]))

        grid = plt.GridSpec(4, 2, wspace=0.5, hspace=0.5)
        fig = plt.figure(figsize=(12, 8), dpi=400)
        ax = plt.subplot(grid[:, :])
        ax.margins(x=0.01)
        ax.tick_params(axis='x', rotation=45)
        ax.plot(self.df['Day Actual'], self.df['Open'].rolling(window=1).mean(), linestyle='-', linewidth=1,
                color='r',
                marker='None')

        ax.plot(x_shift, y_calc, 'black')

        if ax_log:
            ax.set_yscale('log')
            ax.set_ylabel(f'log {self.ticker}', fontsize=10)
        else:
            plt.ylabel(f'{self.ticker}', fontsize=10)

        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        color = cm.rainbow(np.linspace(0, .8, self.s))
        for i, c in zip(range(0, self.s), color):
            if i == 0:
                ax.fill_between(x_shift, y_low[i], y_up[i],
                                color=c, alpha=0.5, label=r'< {} $\sigma$'.format(i + 1))
                ax.fill_between(x_roll, y_low_roll[i], y_up_roll[i],
                                color=c, alpha=0.5)
            else:
                ax.fill_between(x_shift, y_low[i], y_low[i - 1],
                                color=c, alpha=0.5, label=r'< {} $\sigma$'.format(i + 1))
                ax.fill_between(x_roll, y_low_roll[i], y_low_roll[i - 1],
                                color=c, alpha=0.5)
                ax.fill_between(x_shift, y_up[i - 1], y_up[i],
                                color=c, alpha=0.5)
                ax.fill_between(x_shift, y_up_roll[i - 1], y_up_roll[i],
                                color=c, alpha=0.5)

        ax.legend(loc='upper left', prop={'size': 8})
        ax.title.set_text(str(self.df['Date'].iloc[0].strftime('%Y-%m')) + ' to '
                          + str(self.df['Date'].iloc[-1].strftime('%Y-%m')))

        txt = self.fit()['fit_params']
        plt.figtext(0.84, .85, txt, wrap=True, horizontalalignment='left', fontsize=8)

        fig.show()
        if save_fig:
            fig.savefig(f'plots/{ax.yaxis.get_label().get_text()}.png')
