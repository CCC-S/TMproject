import matplotlib.pyplot as plt
import numpy as np
import os

def onEpoch():
    logFile = './log/formal_epoch.log'

    epoch, avgrec = [], []

    with open(logFile, 'r') as f:
        for line in f:
            if 'epoch' in line:
                e = line.split('_')[1]
                epoch.append(int(e))
            if 'AvgRec' in line:
                a = line.strip('\n').split(': ')[1]
                avgrec.append(float(a))

    assert len(epoch) == len(avgrec)

    #print(avgrec)
    #plt.figure()
    plt.title("Effect of Epoch on Average Recall of Cross Validation")
    plt.xlabel("Epoch")
    plt.ylabel("AvgRec")
    #plt.xticks(np.arange(min(epoch), max(epoch)+1, 1.0))
    plt.plot(epoch, avgrec)
    #plt.show()
    plt.savefig('./epoch.png')

def dataDescription():
    labels = ['Test', 'Train']
    atSign = [4905, 14998]
    numSign = [4752, 9218]
    total = [12284, 49570]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, atSign, width, label='At Sign(@)')
    rects2 = ax.bar(x, numSign, width, label='Number Sign(#)')
    rects3 = ax.bar(x + width, total, width, label='Total')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Occurrence')
    ax.set_title('Distribution of Special Characters in Data Set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)


    fig.tight_layout()

    #plt.show()
    plt.savefig("./specialChar.png")

def classifier():

    LR, SGD, LSVC = [], [], []
    x1, x2, x3 = [], [], []

    dire = './log'
    fs = [f for f in os.listdir(dire) if 'acc' in f]
    for f in fs:
        with open(os.path.join(dire, f), 'r') as fin:
            for line in fin:
                x, line = line.strip('\n').split(', ')
                x = float(x)
                line = float(line)
                if 'LR' in f:
                    LR.append(line)
                    x1.append(x)
                elif 'SGD' in f:
                    SGD.append(line)
                    x2.append(x)
                else:
                    LSVC.append(line)
                    x3.append(x)



    fig, ax = plt.subplots()
    ax.set_title("Optimization of Different Classifiers")
    ax.set_xlabel("Epoch of Optimization")
    ax.set_ylabel("performance(AvgRec)")
    #ax.set_xticks(setting_window)

    print()
    ax.plot(x1, LR, 'r-', label="LR")
    ax.plot(x3, LSVC, 'g-', label="LSVC")
    ax.plot(x2, SGD, 'b-', label="SGD")

    # ymax = max(LR_window)
    # xpos = LR_window.index(ymax)
    # xmax = setting_window[xpos]

    # ax.annotate('max point', xy=(xmax, ymax), xytext=(xmax, ymax-0.03),
    #             arrowprops=dict(facecolor='black', shrink=0.0001),
    #             )

    ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(0.75, 0.76))

    #plt.show()
    plt.savefig('./classifiers.png')

classifier()

