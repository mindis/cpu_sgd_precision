import sys, os
import numpy as np;
import matplotlib.pyplot as plt
import struct

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
                vals = line.split('\t');

                if vals[0] == '':
                    o.write("{0}\t-2\t{1}\n".format(id, 1));
                elif int(vals[0]) == 0:
                    o.write("{0}\t-2\t{1}\n".format(id, -1));
                else:
                    o.write("{0}\t-2\t{1}\n".format(id, 1));

                for i in range(1, 14):
                    if vals[i] != '':
                        o.write("{0}\t{1}\t{2}\n".format(id, i-1, int(vals[i])));

                for i in range(14, 40):
                    sval = vals[i].rstrip();
                    if sval != '':
                        #v = struct.unpack('!f', bytes.fromhex(vals[i]))[0]; # FOR PYTHON 3
                        v = struct.unpack('!f', sval.decode('hex'))[0]; # FOR PYTHON 2
                        o.write("{0}\t{1}\t{2:f}\n".format(id, i-1, v));

                id = id+1;

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("You have to specify a file!");
    else:
        process(sys.argv[1]);
