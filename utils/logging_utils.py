verbose = None


def print_verbose(txt):
    global verbose
    if verbose:
        print(txt)