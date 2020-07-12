import pickle


def output_file(obj, filename):
    f = open(filename, "wb")
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def input_file(filename):
    f = open(filename, "rb")
    return pickle.load(f)
