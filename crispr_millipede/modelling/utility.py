import pickle
from datetime import date

'''
    Save a pickle for caching that is notated by the date
'''
def save_or_load_pickle(directory, label, py_object = None, date_string = None):
    if date_string == None:
        today = date.today()
        date_string = str(today.year) + ("0" + str(today.month) if today.month < 10 else str(today.month)) + str(today.day)
    
    filename = directory + label + "_" + date_string + '.pickle'
    print(filename)
    if py_object is None:
        with open(filename, 'rb') as handle:
            py_object = pickle.load(handle)
            return py_object
    else:
        with open(filename, 'wb') as handle:
            pickle.dump(py_object, handle, protocol=pickle.HIGHEST_PROTOCOL)    




from os import listdir
from os.path import isfile, join
'''
    Retrieve all pickles with a label, specifically to identify versions available
'''
def display_all_pickle_versions(directory, label):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and label == f[:len(label)]]