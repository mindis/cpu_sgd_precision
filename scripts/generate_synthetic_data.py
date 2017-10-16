import numpy
import sys
import scipy
from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve
write_to_file = True

def generate_least_squares_data(
        m = 100000,
        # m = 10000000,    # the number of examples
        d = 1000,    # dimension
        sigma = 1, # standard deviation of noise and x_star
        sparsity_fraction = 0.001, # will randomly keep this fraction of the input variable and set everything else to zero,
        suffix = "",
        x_star = None,
        sigma_a = 1
    ):

    extension = ".tsv"
    mu_a = 0
    # mu_a = numpy.random.uniform(-3, 3)


    noise = numpy.random.normal(0, sigma, m) # note that numpy's convention is that
                                             # the scale (the second argument) is std dev, not variance
    if x_star is None:
        x_star = numpy.random.normal(0, sigma, d)

    # make a vector sparse
    def sparsify(vector):
        for index in range(len(vector)):
            if numpy.random.uniform(0, 1) > sparsity_fraction:
                vector[index] = 0

    #base_data_path = '/lfs/raiders4/0/sorathan/hogwild/synthetic'
    base_data_path = '/home/zeke/nethz/cpu_ml/zeke-sgd/data'
    print_m = "1e7"
    if suffix == "_test":
        print_m = "1e6"

    filename = "{base}/synth[m={m}][d={d}][sigma={sigma}][sparsity_fraction={sparsity_fraction}][mu_a={mu_a}][sigma_a={sigma_a}]_sparse{suffix}".format(
        base=base_data_path,
        m=print_m,
        d=d,
        sigma=sigma,
        sparsity_fraction=sparsity_fraction,
        mu_a=mu_a,
        sigma_a=sigma_a,
        suffix=suffix
    )

    print filename + extension
    A = numpy.zeros([d, d])
    v = numpy.zeros(d)
    sum_of_square_b_i = 0 # to calculate loss
    with open(filename + extension, 'w') as f:
        for j in range(m):
            a = numpy.random.normal(0, sigma_a, d)
            sparsify(a)
            a_sparse = scipy.sparse.csr_matrix(a)
            b = a.dot(x_star) + noise[j]
            # update A and v
            # import ipdb; ipdb.set_trace()
            A += numpy.outer(a, a)
            v += a * b
            sum_of_square_b_i += b * b
            # write to file
            if write_to_file:
                # dense format
                # f.write(str(b))
                # for a_i in a:
                #     f.write('\t' + str(a_i))
                # f.write('\n')

                # sparse format
                f.write("{}\t{}\t{}\n".format(j, -2, b))
                for a_i in a_sparse.indices:
                    f.write("{}\t{}\t{}\n".format(j, a_i, a_sparse[0, a_i]))
            else:
                print "{}\t{}\t{}".format(j, -2, b)
                for a_i in a_sparse.indices:
                    print "{}\t{}\t{}".format(j, a_i, a_sparse[0, a_i])

    x_hat = numpy.linalg.solve(A, v)
    print filename + ".meta"
    if write_to_file:
        with open(filename + ".meta", 'w') as f:
            # dense format
            # f.write(str(0)) # Use 0 to encode "x_hat" here
            # for x_hat_i in x_hat:
            #         f.write('\t' + str(x_hat_i))
            # f.write('\n')

            # sparse format
            f.write("{}\t{}\t{}\n".format(0, -2, 0))
            for index in range(len(x_hat)):
                f.write("{}\t{}\t{}\n".format(0, index, x_hat[index]))
    else:
        print "{}\t{}\t{}".format(0, -2, 0)
        for index in range(len(x_hat)):
            print "{}\t{}\t{}".format(0, index, x_hat[index])

    def eval_loss(x):
        return 1/(2.0 * m) * (numpy.dot(numpy.dot(x, A), x) - 2 * numpy.dot(v, x) + sum_of_square_b_i)

    return (x_star, x_hat, eval_loss, m)


if __name__ == '__main__':
    sparsity = float(sys.argv[1])
    (x_star, x_hat, eval_loss, m) = generate_least_squares_data(sparsity_fraction=sparsity)
    print "min loss on train = {value}".format(value=eval_loss(x_hat))
    print "x_hat = {x_hat}".format(x_hat=x_hat)
    print "x_star = {x_star}".format(x_star=x_star)

    (x_star, x_hat_test, eval_loss_test, num_test) = generate_least_squares_data(
        m = 10000,
        suffix="_test",
        sparsity_fraction=sparsity,
        x_star=x_star
    )
    print "x_hat_test = {x_hat_test}".format(x_hat_test=x_hat_test)
    print "min loss on test = {value}".format(value=eval_loss_test(x_hat_test))
