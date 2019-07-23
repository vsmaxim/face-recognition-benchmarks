from .paths import dataset


def normalize_label_from_filename(filename):
    name_parts = filename.split('_')[:-1]
    return '_'.join(name_parts)


def labeled_data_generator():
    for file in dataset.glob('**/*.jpg'):
        label = normalize_label_from_filename(file.name)
        yield file.resolve(), label


def classified_data():
    ret = {}

    for file, label in labeled_data_generator():
        if label not in ret:
            ret[label] = []

        ret[label].append(file)

    return ret
