import os, sys, re, collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

multi_re = re.compile('(?P<prog>[^\s]+) --binary (?P<binary>[01]) --dimension (?P<dimension>[0-9]+) --beta (?P<beta>[0-9.]+) --epoch (?P<epochs>[0-9]+) --batch_size (?P<batch_size>[0-9]+) --stepinitial (?P<stepinitial>[0-9]+) --step_decay (?P<step_decay>[0-9.]+) --quantization (?P<quantization>[0-9]+) --qlevel (?P<qlevel>[0-9]+) --splits (?P<splits>[0-9]+) (?P<train_data>[^\s]+) (?P<test_data>[^\s]+) (?P<meta_data>[^\s]+)')
line_re = re.compile('epoch: (?P<epoch_number>[0-9]+) wall_clock: (?P<wall_clock>[0-9.]+) train_time: (?P<train_time>[0-9.]+) test_time: (?P<test_time>[0-9.]+) epoch_time: (?P<epoch_time>[0-9.]+) compute_time: (?P<compute_time>[0-9.]+) communicate_time: (?P<communicate_time>[0-9.]+) train_loss: (?P<train_loss>[0-9.]+|(inf)) test_loss: (?P<test_loss>[0-9.]+|(inf)) norm_x_minus_x_hat: (?P<norm_x_minus_x_hat>[0-9.]+)\n', re.I)

epochEntry = collections.namedtuple('epochEntry', 'epoch epoch_time total_time, compute_time, communicate_time train_error test_error norm_x_minus_x_hat')
##########################################
# This returns a list of epochEntry (in order by epoch number)
# parsed from the provided file name
# It does basic sanity checking and type conversion
def parse_experiment_file(fname):
    last_epoch = None
    ret_value  = []
    with open(fname, 'r') as f:
        for line in f:
            m = line_re.match(line)
            if m is not None:
                this_epoch = int(m.group('epoch_number'))

                # Sanity Check on epoch parsing to make
                # sure we didn't miss a line
                if last_epoch is None:
                    assert( this_epoch == 0 )
                    last_epoch = this_epoch
                else:
                    assert( (last_epoch + 1) == this_epoch)
                    last_epoch = this_epoch
                
                ret_value.append(
                    epochEntry(
                        epoch=int(m.group('epoch_number')),
                        epoch_time = float(m.group('epoch_time')),
                        total_time = float(m.group('wall_clock')),
                        compute_time = float(m.group('compute_time')),
                        communicate_time = float(m.group('communicate_time')),
                        train_error= float(m.group('train_loss')),
                        test_error = float(m.group('test_loss')),
                        norm_x_minus_x_hat = float(m.group('norm_x_minus_x_hat'))
                    )
                )

    return ret_value

def parse_experiment_file_multiple(fname):

    vals = [];
    info = '';
    nof_nodes = '';
    
    cur_name = '';
    last_epoch = None
    ret_value  = []
    with open(fname, 'r') as f:
        next(f);
        nof_nodes = next(f);
        for line in f:
            m = multi_re.match(line);
            if m is not None:
                if cur_name != '':
                    vals.append((cur_name, ret_value));
                    ret_value = [];
                    last_epoch = None;
                else:
                    info = '{0}Dataset: {1}'.format(nof_nodes, m.group('train_data').rsplit('/',1)[-1]);

                cur_name = '{0}'.format(m.group('prog').rsplit('/',1)[-1]);

            elif cur_name != '':
                m = line_re.match(line)
                if m is not None:
                    this_epoch = int(m.group('epoch_number'))

                    # Sanity Check on epoch parsing to make
                    # sure we didn't miss a line
                    if last_epoch is None:
                        assert( this_epoch == 0 )
                        last_epoch = this_epoch
                    else:
                        assert( (last_epoch + 1) == this_epoch)
                        last_epoch = this_epoch
                    
                    ret_value.append(
                        epochEntry(
                            epoch=int(m.group('epoch_number')),
                            epoch_time = float(m.group('epoch_time')),
                            total_time = float(m.group('wall_clock')),
                            compute_time = float(m.group('compute_time')),
                            communicate_time = float(m.group('communicate_time')),
                            train_error= float(m.group('train_loss')),
                            test_error = float(m.group('test_loss')),
                            norm_x_minus_x_hat = float(m.group('norm_x_minus_x_hat'))
                        )
                    )
    if cur_name != '':
        vals.append((cur_name, ret_value));

    return (vals, info);

def process_run_nprocesses(path):
    vals = [];

    file_re = re.compile('linreg\[p=(?P<nprocesses>[0-9]+)\]\[t=(?P<nthreads>[0-9]+)\]');
    take_re = re.compile('\[take=(?P<takenr>[0-9]+)\]');

    if not os.path.isdir(path):
        print("Folder '{0}' not found!".format(path));
        return vals;

    print("Processing folder '{0}'...".format(path));

    for run in os.listdir(path):
        m_filereg = file_re.match(run);
        if m_filereg is not None:
            nprocesses = int(m_filereg.group('nprocesses'));
            all_takes = [];

            for f in os.listdir(path + '/' + run):
                m_takere = take_re.match(f);
                if m_takere is not None:
                    takenr = int(m_takere.group('takenr'));
                    entries = parse_experiment_file(path + "/" + run + "/" + f);
                    all_takes.append((takenr, filter(lambda x: x.epoch != 0, entries)));

            vals.append((nprocesses, all_takes));

    vals.sort(key=lambda x: x[0]);
    return vals;

def process_run_batchsize(path):
    vals = [];

    file_re = re.compile('linreg\[b=(?P<batchsize>[0-9]+)\]');
    take_re = re.compile('\[take=(?P<takenr>[0-9]+)\]');

    if not os.path.isdir(path):
        print("Folder '{0}' not found!".format(path));
        return vals;

    print("Processing folder '{0}'...".format(path));

    for run in os.listdir(path):
        m_filereg = file_re.match(run);
        if m_filereg is not None:
            batchsize = int(m_filereg.group('batchsize'));
            all_takes = [];

            for f in os.listdir(path + '/' + run):
                m_takere = take_re.match(f);
                if m_takere is not None:
                    takenr = int(m_takere.group('takenr'));
                    entries = parse_experiment_file(path + "/" + run + "/" + f);
                    all_takes.append((takenr, filter(lambda x: x.epoch != 0, entries)));

            vals.append((batchsize, all_takes));

    vals.sort(key=lambda x: x[0]);
    return vals;

def process_single_file(path):
    if not os.path.exists(path):
        print("File '{0}' not found!".format(path));
        return (None, None);

    return parse_experiment_file_multiple(path);

def process_run_communication_computation(path):
    vals = [];

    file_re = re.compile('\[(?P<dataset>[0-9a-zA-Z_]+)\]\[(?P<strategy>[0-9a-zA-Z_]+)\]\[(?P<stepsize>[0-9a-zA-Z_]+)\]\[(?P<representation>[0-9a-zA-Z_]+)\]');

    if not os.path.isdir(path):
        print("Folder '{0}' not found!".format(path));
        return vals;

    print("Processing folder '{0}'...".format(path));

    for f in os.listdir(path):
        m_filereg = file_re.match(f);
        if m_filereg is not None:
            print("Processing file '{0}'...".format(f));
            dataset = m_filereg.group('dataset');
            strategy = m_filereg.group('strategy');
            stepsize = m_filereg.group('stepsize');
            representation = m_filereg.group('representation');
            label = "{0}-{1}-{2}-{3}".format(dataset, strategy, stepsize, representation);
            entries = parse_experiment_file(path + "/" + f);
            if len(entries) > 0:
                vals.append((label, filter(lambda x: x.epoch != 0, entries)));

    vals.sort(key=lambda x: x[0]);
    return vals;

def avgReduceFunc(values, extractFunc):
    values = map(lambda x: (x[0], map(lambda y: map(extractFunc, y[1]), x[1])), values);
    values = map(lambda x: (x[0], map(lambda y: float(sum(y)/len(y)), x[1])), values);
    return values;

def lastReduceFunc(values, extractFunc):
    values = map(lambda x: (x[0], map(lambda y: map(extractFunc, y[1]), x[1])), values);
    values = map(lambda x: (x[0], map(lambda y: y[-1], x[1])), values);
    return values;

def extract_val(values, extractFunc, reduceFunc):
    values = reduceFunc(values, extractFunc);

    return ([x[0] for x in values], [x[1] for x in values]);

def extract_norm_x_minus_x_hat(epochEntry):
    return epochEntry.norm_x_minus_x_hat;

def extract_epochtimes(epochEntry):
    return epochEntry.epoch_time;

def extract_avg_computetimes(epochEntry):
    return epochEntry.compute_time;

def extract_avg_communicatetimes(epochEntry):
    return epochEntry.communicate_time;

def extract_avg_trainerror(epochEntry):
    return epochEntry.train_error;

def extract_avg_testerror(epochEntry):
    return epochEntry.test_error;

def createSingleFilePlot(values, labels, valFunc, xlabel, ylabel, title, plotToFile, saveFileName):
    colors = ['b', 'g', 'r', 'k', 'y', 'c', 'm'];
    colorIndex = 0;

    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(111)

    xmax = 0;
    ymax = 0;
    for i, label in enumerate(labels):
        (x, y) = (map(lambda x: x.epoch, values[i]), map(lambda x: valFunc[i](x), values[i]));
        if x is not None and len(x) > 0:
            ax.plot(x, y, colors[colorIndex], label=label);
        xmax = max(xmax, max(x));
        ymax = max(ymax, max(y));
        colorIndex = (colorIndex + 1 % len(colors));

    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel);
    #ax.set_axis([0, xmax*1.01, 0, ymax*1.2]);
    ax.set_title(title);
    ax.legend();
    plt.tight_layout()

    if(plotToFile):
        plt.savefig(saveFileName);
    else:
        plt.show();


def createPlot(values, labels, valueFromEpochEntryFuncs, reduceFunc, xlabel, ylabel, title, plotToFile, saveFileName):
    plt.figure();

    colors = ['b', 'g', 'r'];
    colorIndex = 0;

    xmax = 0;
    ymax = 0;
    for i, label in enumerate(labels):
        (x, y) = extract_val(values[i], valueFromEpochEntryFuncs[i], reduceFunc);
        avg = map(lambda v: float(sum(v)/len(v)), y);
        minVals = map(lambda v: min(v), y);
        maxVals = map(lambda v: max(v), y);
        if x is not None and len(x) > 0:
            plt.plot(x, avg, colors[colorIndex], label=label);
            plt.fill_between(x, minVals, maxVals, color=colors[colorIndex], alpha=0.1);
        xmax = max(xmax, max(x));
        ymax = max(ymax, max(maxVals));
        colorIndex = (colorIndex + 1 % len(colors));

    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    #plt.axis([0, xmax*1.01, 0, ymax*1.2]);
    plt.title(title);
    plt.legend();
    if(plotToFile):
        plt.savefig(saveFileName);
    else:
        plt.show();

def createBarCompCommTimePlot(values, title, plotToFile, saveFileName):

    names = ["MPI AllReduce", "SAllReduce (RecDoubling)", "SAllReduce (Split, Dense)", "SAllReduce (Split)"]
    ylabels = map(lambda x: names[int(x[0])], values);
    #ylabels = map(lambda x: x[0], values);
    xcommunicate_time = map(lambda x: map(lambda y: y.communicate_time, x[1]), values);
    xcompute_time = map(lambda x: map(lambda y: y.compute_time, x[1]), values);
    xcommunicate_time = map(lambda x: float(sum(x)/len(x)), xcommunicate_time);
    xcompute_time = map(lambda x: float(sum(x)/len(x)), xcompute_time);

    data = np.stack((xcompute_time, xcommunicate_time));
    label = ["Computation time", "Communication time"];

    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(111)
    y_pos = np.arange(len(ylabels))

    colors ='bgr'
    patch_handles = []
    left = np.zeros(len(ylabels)) # left alignment of data starts at zero
    for i, d in enumerate(data):
        patch_handles.append(ax.barh(y_pos, d, 
                    color=colors[i%len(colors)], align='center', 
                    left=left, label=label[i]))
        # accumulate the left-hand offsets
        left += d

    font = {'family' : 'normal',
            'size'   : 30}
    matplotlib.rc('font', **font)
    #ax.set_title(title);
    ax.set_yticks(y_pos);
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('Time (s)')
    ax.legend();

    plt.tight_layout()
    if(plotToFile):
        plt.savefig(saveFileName);
    else:
        plt.show();

def process_experiment(experiment_nr, plotToFile, dataset, stepsizemethod, plotMidTitle, filePrefix):
    if(plotToFile):
        matplotlib.use('pdf');

    folder = "results/{0}/{1}/2_parameter_server_async/{2}".format(experiment_nr, dataset, stepsizemethod);
    paramServerAsyncValues = process_run_nprocesses(folder);
    if(paramServerAsyncValues is None or len(paramServerAsyncValues) == 0):
        print("No values found in folder '{0}'!".format(folder));
        return;

    folder = "results/{0}/{1}/3_all_reduce/{2}".format(experiment_nr, dataset, stepsizemethod);
    allReduceValues = process_run_nprocesses(folder);
    if(allReduceValues is None or len(allReduceValues ) == 0):
        print("No values found in folder '{0}'!".format(folder));
        return;

    createPlot([allReduceValues], 
        ["All Reduce - Training"],
        [extract_avg_trainerror],
        lastReduceFunc,'# of processes' , 'Error',
        "Exp. # '{0}' - {1} - Last Train Error".format(experiment_nr, plotMidTitle),
        plotToFile,
        "results/{0}/{1}last_allreduce_trainerror.pdf".format(experiment_nr, filePrefix));
    
    createPlot([paramServerAsyncValues, allReduceValues],
            ["ParameterServer Async", "All Reduce"],
            [extract_norm_x_minus_x_hat, extract_norm_x_minus_x_hat],
            lastReduceFunc, '# of processes' , 'x minux x_hat',
            "Exp. # '{0}' - {1} - x minus x_hat".format(experiment_nr, plotMidTitle),
            plotToFile,
            "results/{0}/{1}last_x_minus_x_hat.pdf".format(experiment_nr, filePrefix));

    createPlot([paramServerAsyncValues, allReduceValues],
            ["ParameterServer Async", "All Reduce"],
            [extract_epochtimes, extract_epochtimes],
            avgReduceFunc,'# of processes' , 'average time (s)',
            "Exp. # '{0}' - {1} - Avg Epoch Time".format(experiment_nr, plotMidTitle),
            plotToFile,
            "results/{0}/{1}avg_epoch_time.pdf".format(experiment_nr, filePrefix));

    createPlot([paramServerAsyncValues, allReduceValues], 
        ["ParameterServer Async", "All Reduce"],
        [extract_avg_computetimes, extract_avg_computetimes],
        avgReduceFunc,'# of processes' , 'average time (s)',
        "Exp. # '{0}' - {1} - Avg Computation Time".format(experiment_nr, plotMidTitle),
        plotToFile,
        "results/{0}/{1}avg_computation_time.pdf".format(experiment_nr, filePrefix));

    createPlot([paramServerAsyncValues, allReduceValues], 
        ["ParameterServer Async", "All Reduce"],
        [extract_avg_communicatetimes, extract_avg_communicatetimes],
        avgReduceFunc,'# of processes' , 'average time (s)',
        "Exp. # '{0}' - {1} - Avg Communication Time".format(experiment_nr, plotMidTitle),
        plotToFile,
        "results/{0}/{1}avg_communication_time.pdf".format(experiment_nr, filePrefix));

    createPlot([paramServerAsyncValues, allReduceValues], 
        ["ParameterServer Async", "All Reduce"],
        [extract_avg_trainerror, extract_avg_trainerror],
        lastReduceFunc,'# of processes' , 'Training error',
        "Exp. # '{0}' - {1} - Last Training Error".format(experiment_nr, plotMidTitle),
        plotToFile,
        "results/{0}/{1}last_trainerror.pdf".format(experiment_nr, filePrefix));

    createPlot([paramServerAsyncValues, allReduceValues], 
        ["ParameterServer Async", "All Reduce"],
        [extract_avg_testerror, extract_avg_testerror],
        lastReduceFunc,'# of processes' , 'Testing error',
        "Exp. # '{0}' - {1} - Last Testing Error".format(experiment_nr, plotMidTitle),
        plotToFile,
        "results/{0}/{1}last_testerror.pdf".format(experiment_nr, filePrefix));

def plotTrainLoss(experiment_nr, dataset, stepsizemethod, plotToFile, filePrefix):
    if(plotToFile):
        matplotlib.use('pdf');

    folder = "results/{0}/{1}/3_all_reduce/{2}".format(experiment_nr, dataset, stepsizemethod);
    allReduceValues = process_run_batchsize(folder);
    if(allReduceValues is None or len(allReduceValues ) == 0):
        print("No values found in folder '{0}'!".format(folder));
        return;

    createPlot([allReduceValues], 
        ["All Reduce"],
        [extract_avg_trainerror],
        lastReduceFunc, 'Batch size', 'Error',
        "Exp. # '{0}' - Train Error after 20 epochs".format(experiment_nr),
        plotToFile,
        "results/{0}/{1}allreduce_last_trainerror.pdf".format(experiment_nr, filePrefix));

def plotBarCompCommTime(experiment_nr, plotToFile):
    folder = "results/{0}/".format(experiment_nr);

    values = process_run_communication_computation(folder);

    createBarCompCommTimePlot(values, "Avg. computation & communication time", plotToFile, "results/{0}/bar_communication_computation_plot.pdf".format(experiment_nr));

def processSingleFile(experiment_nr, plotToFile):
    folder = "results/{0}/output.log".format(experiment_nr);

    (values, info) = process_single_file(folder);
    if values is None:
        print("No values found in file!");
        return;

    createBarCompCommTimePlot(values, "Avg. epoch time - {0}".format(info), plotToFile, "results/{0}/bar_communication_computation_plot.pdf".format(experiment_nr));

    createSingleFilePlot([x[1] for x in values],
        [x[0] for x in values],
        len(values)*[extract_avg_trainerror],
        'Epoch' , 'Error',
        "Exp. # '{0}' - Train Error - {1}".format(experiment_nr, info),
        plotToFile,
        "results/{0}/trainerror.pdf".format(experiment_nr));

    #createSingleFilePlot([x[1] for x in values],
    #    [x[0] for x in values],
    #    len(values)*[extract_avg_testerror],
    #    'Epoch' , 'Error',
    #    "Exp. # '{0}' - Test Error - {1}".format(experiment_nr, info),
    #    plotToFile,
    #    "results/{0}/testerror.pdf".format(experiment_nr));

    # Convergence speedup
    for epoch in range(0, len(values[-1][1])):
        val = values[-1][1][epoch].train_error;
        for i in range(0,len(values)):
            values[i][1][epoch] = values[i][1][epoch]._replace(train_error=values[i][1][epoch].train_error / val);

    createSingleFilePlot([x[1] for x in values],
        [x[0] for x in values],
        len(values)*[extract_avg_trainerror],
        'Epoch' , 'Ratio',
        "Exp. # '{0}' - Train Error (Convergence Slowdown) - {1}".format(experiment_nr, info),
        plotToFile,
        "results/{0}/trainerror_slowdown.pdf".format(experiment_nr));


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("You have to specify the experiment number!");
    else:
        plotToFile = False;
        # process_experiment( sys.argv[1], plotToFile, "1_synthetic", "1_exp_backoff_stepsizes", "Linear Regression (EXPBACKOFF)", "LinReg_ExpBackOff_" )
        # process_experiment( sys.argv[1], plotToFile, "2_higgs", "1_exp_backoff_stepsizes", "Logistic Regression (EXPBACKOFF)", "LogReg_ExpBackOff_" )
        # process_experiment( sys.argv[1], plotToFile, "1_synthetic", "2_decreasing_stepsizes", "Linear Regression (DECREASING)", "LinReg_Decreasing_" )
        # process_experiment( sys.argv[1], plotToFile, "2_higgs", "2_decreasing_stepsizes", "Logistic Regression (DECREASING)", "LogReg_Decreasing_" )
        # plotTrainLoss( sys.argv[1], "2_higgs", "1_exp_backoff_stepsizes", plotToFile, "LogReg_ExpBackOff_BatchSize_");
        # plotBarCompCommTime( sys.argv[1], plotToFile);
        processSingleFile( sys.argv[1], plotToFile);
