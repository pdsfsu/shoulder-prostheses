from decimal import *

# autoResult is the resulting image from automatic segmentation
# manResult is the resulting image from manual segmentation (i.e., the ground truth image)
# mask is an optional mask to be applied to manResult
# Returns (true positives, false positives, true negatives, false negatives)

decimal_precision = 2

def findTotals(autoResult, manResult):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

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

#  For Otsu with 3 thresholds and more intensity values in autoResult
def findTotalsOtsu3(autoResult, manResult, intensity):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

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
            if pixelA == intensity and pixelB > 0:
                tp += 1
            elif pixelA == intensity and pixelB == 0:
                fp += 1
            elif pixelA != intensity and pixelB == 0:
                tn += 1
            else:
                fn += 1

    return (tp, fp, tn, fn)

# Returns precision, which is the number of correct results divided by the number of all returned results
def precision (truePos, falsePos):
    global decimal_precision
    getcontext().prec = decimal_precision
    tp = Decimal(truePos)
    fp = Decimal(falsePos)
    return tp / (tp + fp)

# Returns sensitivity, which is the number of correct results divided by the number of results that should have been
# returned
def sensitivity(truePos, falseNeg):
    global decimal_precision
    getcontext().prec = decimal_precision
    tp = Decimal(truePos)
    fn = Decimal(falseNeg)
    return tp / (tp + fn)

def fmeasure(truePos, falsePos, falseNeg):
    global decimal_precision
    getcontext().prec = decimal_precision
    tp = Decimal(truePos)
    fp = Decimal(falsePos)
    fn = Decimal(falseNeg)
    p = precision(tp, fp)
    s = sensitivity(tp, fn)
    return 2 * (p * s) / (p + s)

def dicecoeff(truePos, falsePos, falseNeg):
    global decimal_precision
    getcontext().prec = decimal_precision
    tp = Decimal(truePos)
    fp = Decimal(falsePos)
    fn = Decimal(falseNeg)
    return (2 * tp) / ( (2 * tp) + fp + fn)

def jaccardindex(truePos, falsePos, falseNeg):
    global decimal_precision
    getcontext().prec = decimal_precision
    tp = Decimal(truePos)
    fp = Decimal(falsePos)
    fn = Decimal(falseNeg)
    return tp / (tp + fp + fn)