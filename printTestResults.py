from prettytable import PrettyTable
import evaluation
from datetime import datetime


# lines in file should have the following formant:
#   version tp fp tn fn
def circle_results(file):
    t = PrettyTable(['Description', 'Total', 'True Pos.', 'False Pos.', 'True Neg.', 'False Neg.', 'Precision', 'Sensitivity',
                     'Dice Coeff.', 'Jaccard Ind.', 'Jaccard Dist.'])
    f = open(file, 'r')
    for line in f:
        totals = line.split()
        description = totals[0]
        tp = totals[1]
        fp = totals[2]
        tn = totals[3]
        fn = totals[4]
        total = int(tp) + int(fp) + int(tn) + int(fn)
        precision = evaluation.precision(tp, fp)
        sensitivity = evaluation.sensitivity(tp, fn)
        # fmeasure = evaluation.fmeasure(tp, fp, fn)
        dicecoeff = evaluation.dicecoeff(tp, fp, fn)
        jaccardindex = evaluation.jaccardindex(tp, fp, fn)
        jaccarddifference = 1 - jaccardindex
        t.add_row([description, str(total), str(tp), str(fp), str(tn), str(fn), str(precision), str(sensitivity), str(dicecoeff),
                  str(jaccardindex), str(jaccarddifference)])

    print "Circle Detection\n"
    print t

    now = "\n" + str(datetime.now()) + "\n"
    data = t.get_string()
    r = open('circle_test_results.txt', 'ab')
    r.write(now)
    r.write(data)
    r.write("\n")
    r.close()

# test
circle_results('circle_detection.txt')