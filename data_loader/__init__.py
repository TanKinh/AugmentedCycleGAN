
def load_font2hand(root):
    """loads in memory numpy data files"""
    def _load(fname):
        arr = np.load(os.path.join(root, fname))
        # convert data from b,0,1,c to b,c,0,1
        arr = np.transpose(arr, (0,3,1,2))
        # scale and shift to [-1,1]
        arr = arr / 127.5 - 1.
        return arr.astype('float32')

    print ("loading data numpy files...")
    trainA = _load("trainA.npy")
    trainB = _load("trainB.npy")
    testA  = _load("valA.npy")
    testB  = _load("valB.npy")
    print ("done.")

    # shuffle train data
    # rand_state = random.getstate()
    # random.seed(123)
    # indx = range(len(trainA))
    # random.shuffle(indx)
    # trainA = trainA[indx]
    # trainB = trainB[indx]
    # random.setstate(rand_state)

    devA = trainA[:DEV_SIZE]
    devB = trainB[:DEV_SIZE]

    trainA = trainA[DEV_SIZE:]
    trainB = trainB[DEV_SIZE:]

    return trainA, trainB, devA, devB, testA, testB