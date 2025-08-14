import pickle

with open('/Storage/ying/pyCortexProj/scripts/test_X.pickle', 'rb') as handle:
    b = pickle.load(handle)
    print(b)

