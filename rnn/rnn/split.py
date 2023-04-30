import sys
import random


def split_and_write(infile, train, valid, test):
    header = infile.readline()
    train.write(header)
    valid.write(header)
    test.write(header)

    for line in infile:
        rand = random.uniform(0.0, 1.0)

        stream = test if rand < 0.2 else (valid if rand < 0.3 else train)
        stream.write(line)


def split(ds, train, valid, test):
    with open(ds) as infile:
        with open(train, 'w') as trs:
            with open(valid, 'w') as vs:
                with open(test, 'w') as tss:
                    split_and_write(infile, trs, vs, tss)


if __name__ == '__main__':
    split(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
