import sys


def count_categories(file):
    cats = dict()

    with open(file) as infile:
        infile.readline()
        for line in infile:
            last = line[-2]
            cats[last] = cats.get(last, 0) + 1

    return cats


def main(file):
    cats = count_categories(file)
    print('| Rating | Count |')
    print('|--------|-------|')
    for r in sorted(cats):
        print(f'| {r} | {cats[r]} |')


if __name__ == '__main__':
    main(sys.argv[1])
