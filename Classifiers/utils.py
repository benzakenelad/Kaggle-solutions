def readData(filename):
    file = open(filename, 'r')
    samples = file.readlines()
    n = len(samples)
    for i in range(n):
        samples[i] = samples[i].replace('\n', '')
        samples[i] = samples[i].split('\t')
    d = len(samples[0]) - 1
    attributes = samples.pop(0)[:d]
    X = [x[:d] for x in samples]
    y = [x[d] for x in samples]
    return X, y, attributes


def calc_accuracy(predictions, labels):
    if len(predictions) != len(labels):
        raise ValueError('predictions len and labels len are not equal')
    n = len(predictions)
    acc = 0
    for i in range(n):
        if predictions[i] == labels[i]:
            acc += 1
    return acc / n


def extract_unique_labels_and_labels_counter(vector):
    label_values = set()
    for label in vector:
        label_values.add(label)
    label_values = list(label_values)
    label_map = {k: v for v, k in enumerate(label_values)}
    labels_counter = [0] * len(label_values)
    for label in vector:
        labels_counter[label_map.get(label)] += 1

    return label_values, labels_counter


def argmax(values):
    return max(enumerate(values), key=lambda x: x[1])[0]


def argsort(values):
    return sorted(range(len(values)), key=values.__getitem__)


def remove_column(data, col_ind):
    d = len(data[0])
    n = len(data)
    new_data = []
    for i in range(n):
        new_data.append([data[i][j] for j in range(d) if j != col_ind])
    return new_data


