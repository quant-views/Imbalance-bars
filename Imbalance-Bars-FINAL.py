## Add comment here ##
def delta(df):
    a = np.diff(df['Price'])
    a = np.insert(a, 0, 0)
    df['Delta'] = a
    return df

def labelling(df):
    b = np.ones(len(df['Price']))
    for i, delta in enumerate(df['Delta']):
        if i > 0:
            if delta == 0:
                b[i] = b[i-1]
            else:
                b[i] = abs(delta) / delta
    df['Label'] = b
    return df

def initial_conditions(df):
    prob = pd.DataFrame(pd.pivot_table(df, index='Label', values='Price', aggfunc='count'))
    prob = np.array(prob)
    p_b = prob[1]/(prob[0]+prob[1])
    p_s = prob[0]/(prob[0]+prob[1])
    return p_b, p_s

def bar_gen_run(df, thresh):
    cumm, open, low, high, close, cumm_vol, vol_price, b, s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    collector, bar, thresh_buffer = [], [], []

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):
        if label == 1:
            b = b + label
        else:
            s = s + label
        theta = max(b, abs(s))

        cumm_vol = cumm_vol + volume
        vol_price = vol_price + (price * volume)
        collector.append(price)
        if theta >= thresh:
            open = collector[0]
            high = np.max(collector)
            low = np.min(collector)
            close = collector[-1]
            vwap = vol_price / cumm_vol
            bar.append((date, i, open, low, high, close, vwap))
            a = len(collector) * max(((b/len(collector)), (1-(b/len(collector)))))
            thresh_buffer.append(a)
            if i > 500000: thresh = np.average(thresh_buffer)
            theta, open, low, high, close, cumm_vol, vol_price, b, s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            collector = []
    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    run_bar = pd.DataFrame(bar, columns= cols)
    return run_bar

def bar_gen(df, thresh):
    cumm, open, low, high, close, cumm_vol, vol_price, b = 0, 0, 0, 0, 0, 0, 0, 0
    collector, bar, thresh_buffer = [], [], []

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):
        if label == 1:
            b = b + 1
        cumm = cumm + label
        cumm_vol = cumm_vol + volume
        vol_price = vol_price + (price * volume)
        collector.append(price)
        if abs(cumm) >= thresh:
            open = collector[0]
            high = np.max(collector)
            low = np.min(collector)
            close = collector[-1]
            vwap = vol_price / cumm_vol
            bar.append((date, i, open, low, high, close, vwap))
            a = len(collector) * abs((2*(b/len(collector)))-1)
            thresh_buffer.append(a)
            if i > 500000: thresh = np.average(thresh_buffer)
            cumm, open, low, high, close, cumm_vol, vol_price, b = 0, 0, 0, 0, 0, 0, 0, 0
            collector = []
    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    imbalance_bar = pd.DataFrame(bar, columns= cols)
    return imbalance_bar


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf
    import seaborn as sns
    import scipy.stats as stats

    # Import Data
    df = pd.read_csv(r'C:\Users\josde\PycharmProjects\Imbalance-bars\ES_Trades.csv')
    df = df.iloc[:, 0:5]
    df['Dollar'] = df['Price']*df['Volume']

    # Price change & Labeling
    df = delta(df)
    df = labelling(df)

    # Initial conditions
    p_b, p_s = initial_conditions(df)
    thresh_imbalance = 800
    thresh_run = 3200

    # Generate imbalance bars
    imbalance_bar = bar_gen(df, thresh_imbalance)
    print(imbalance_bar)

    # Generate run bars
    run_bar = bar_gen_run(df, thresh_run)
    print(run_bar)

    # Plot bars
    plt.figure(1)
    plt.plot(imbalance_bar['Vwap'],'r')
    plt.plot(run_bar['Vwap'],'g')
    plt.show()