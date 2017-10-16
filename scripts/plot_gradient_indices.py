import os, sys, re, collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

printToFile = False;
reorder = False;

def plot_gradient_indices(fname, dim, splits):

    indices = range(0,dim);
    data = np.zeros(dim);
    sumDat = 0;
    #data = [];

    with open(fname) as input_file:
        for line in input_file:
            line = line.strip();
            nums = line.split();
            if len(nums) >= 2:
                index = int(nums[1]);
                data[index] += 1;
                sumDat += 1;
                #data.append(int(nums[1]));

    if reorder:
        data = sorted(data, reverse=True);

    if splits > 0:
        output = "Sizes: ";
        idx = 0;
        step = int(sumDat / splits);
        tmp = 0;
        prev = 0;
        for i in range(0, dim):
            if tmp > step:
                tmp = 0;
                print("Process {0}: {1}".format(idx, prev));
                output += "{0}, ".format(i - prev);
                idx += 1;
                prev = i;
            tmp += data[i];
        output += "{0}".format(dim - prev);
        print(output);

    plt.figure();
    #print(len(indices));
    #print(len(data));
    #print(min(data));
    #print(max(data));
    plt.plot(indices, data, 'b+');
    #plt.hist(data);

    if printToFile:
        output = os.path.splitext(fname)[0] + ".pdf";
        plt.savefig(output);
    else:
        plt.show()

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("You have to specify the filename containing the gradients and the dimension of the data");
    elif (len(sys.argv) == 4):
        plot_gradient_indices(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]));
    else:
        plot_gradient_indices(sys.argv[1], int(sys.argv[2]), -1);
