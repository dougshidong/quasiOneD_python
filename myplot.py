import matplotlib.pyplot as plt

def plot(x, y, fig_id=None, title='', lab='', ymarker='o'):
    plt.figure(fig_id)
    plt.plot(x, y, label = lab, marker = ymarker)
    if title!='': 
        plt.title(title)
    if lab!='': 
        plt.legend()

def plot_compare(x, current, target, fig_i=None, title_i=''):
    plot(x, current, fig_id=fig_i, title=title_i, lab='Current')
    plot(x, target,  fig_id=fig_i, title=title_i, lab='Target')

