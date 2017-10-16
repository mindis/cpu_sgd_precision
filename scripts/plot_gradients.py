import os, sys, re, collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

printToFile = True;

def update_hist(num, data, xmin, xmax):
    plt.cla();
    plt.title("Batch {0}".format(num));
    plt.hist(data[num]);
    plt.xlim(xmin, xmax);

def plot_gradients(fname):

    data = dict();

    with open(fname) as input_file:
        for line in input_file:
            line = line.strip();
            nums = line.split();
            if len(nums) >= 2:
                frame = int(nums[0]);
                if frame not in data:
                    data[frame] = [];
                data[frame].append(float(nums[2]));

    xmin = min(map(lambda x: min(x), data.values()));
    xmax = max(map(lambda x: max(x), data.values()));

    fig = plt.figure();
    update_hist(min(data.keys()), data, xmin, xmax);

    anim = animation.FuncAnimation(fig, update_hist, data.keys(), fargs=(data, xmin, xmax, ), repeat=False )
    if printToFile:
        output = os.path.splitext(fname)[0] + ".gif";
        anim.save(output, writer='imagemagick', fps=5)
    else:
        plt.show()

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("You have to specify the filename containing the gradients");
    else:
        plot_gradients(sys.argv[1]);
