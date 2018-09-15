import matplotlib.pyplot as plt

def plot(x, y, fig_id=None, title=None, lab=None, ymarker='o'):
    plt.figure(fig_id)
    plt.title(title)
    plt.plot(x, y, label = lab, marker = ymarker)
    plt.legend()

