
# for f in *.csv; do
#   python playlist.py "$f" "${f%.csv}list.txt"
# done

# autoResult is the resulting image from automatic segmentation
# manResult is the resulting image from manual segmentation (i.e., the ground truth image)
# mask is an optional mask to be applied to manResult
# Returns (true positives, false positives, true negatives, false negatives)

def findTotals(autoResult, manResult):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # x, y = (image > limit).nonzero()
    # vals = image[x, y]

    if autoResult.shape != manResult.shape:
        print "Mismatch"
        f = open('Errors.txt', 'a')
        f.write("autoResult shape: " + str(autoResult.shape) + " manResult shape: " + str(manResult.shape) + "\n")
        f.close()
        return (0,0,0,0)

    for i in xrange(autoResult.shape[0]):
        for j in xrange(autoResult.shape[1]):
            pixelA = autoResult.item(i, j)
            pixelB = manResult.item(i, j)
            if pixelA > 0 and pixelB > 0:
                tp += 1
            elif pixelA > 0 and pixelB == 0:
                fp += 1
            elif pixelA == 0 and pixelB == 0:
                tn += 1
            else:
                fn += 1

    return (tp, fp, tn, fn)

# Returns precision, which is the number of correct results divided by the number of all returned results
def precision (truePos, falsePos):
    tp = float(truePos)
    fp = float(falsePos)
    return tp / (tp + fp)

# Returns sensitivty, which is the number of correct results divided by the number of results that should have been
# returned
def sensitivity(truePos, falseNeg):
    tp = float(truePos)
    fn = float(falseNeg)
    return tp / (tp + fn)

def fmeasure(truePos, falsePos, falseNeg):
    tp = float(truePos)
    fp = float(falsePos)
    fn = float(falseNeg)
    p = precision(tp, fp)
    s = sensitivity(tp, fn)

    return 2 * (p * s) / (p + s)

def dicecoeff(truePos, falsePos, falseNeg):
    tp = float(truePos)
    fp = float(falsePos)
    fn = float(falseNeg)
    return (2 * tp) / ( (2 * tp) + fp + fn)

def jaccardindex(truePos, falsePos, falseNeg):
    tp = float(truePos)
    fp = float(falsePos)
    fn = float(falseNeg)
    return tp / (tp + fp + fn)