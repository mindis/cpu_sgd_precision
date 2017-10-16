import sys, os
import numpy as np;
import matplotlib.pyplot as plt

def process(filename):

    if not os.path.exists(filename):
        print("File '{0}' does not exists!".format(filename));
        return;

    output = os.path.splitext(filename)[0] + ".tsv";
    print("Writing to file: " + output);

    id=0;
    with open(filename, 'r') as f:
        with open(output, 'w') as o:
            for line in f:
                vals = line.split();
                o.write("{0}\t-2\t{1}\n".format(id, vals[0]));
                for pair in vals[1:]:
                    s = pair.split(':');
                    o.write("{0}\t{1}\t{2}\n".format(id, s[0], s[1]));
                id = id+1;

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("You have to specify a file!");
    else:
        process(sys.argv[1]);
