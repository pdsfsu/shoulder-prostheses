import evaluation
from prettytable import PrettyTable
from datetime import datetime


# files = ['b.txt', 'bh.txt', 'n.txt', 'o1.txt', 'o3_t3-max.txt']
# afiles = ['b_all.txt', 'bh_all.txt','n_all.txt', 'o1_all.txt', 'o3_t3-max_all.txt']
files = ['b.txt', 'n.txt', 'o1.txt']
afiles = ['b_all.txt', 'n_all.txt', 'o1_all.txt']
now = "\n" + str(datetime.now()) + "\n"
r = open('segmentation_test_results.txt', 'a')
r.write(now)
r.write("\nAll Segmentation Results \n")
r.close()

t = PrettyTable(['Version', 'Total', 'True Pos.', 'False Pos.', 'True Neg.', 'False Neg.', 'Precision', 'Sensitivity',
                 'Dice Coeff.', 'Jaccard Ind.', 'Jaccard Dist.'])
for afile in afiles:
    lines = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    af = open(afile, 'r')
    for line in af:
        lines += 1
        totals = line.split()
        # totals[0] is image name
        tp += int(totals[1])
        fp += int(totals[2])
        tn += int(totals[3])
        fn += int(totals[4])

    total = tp + fp + tn + fn
    af.close()

    precision = evaluation.precision(tp, fp)
    sensitivity = evaluation.sensitivity(tp, fn)
    fmeasure = evaluation.fmeasure(tp, fp, fn)
    dicecoeff = evaluation.dicecoeff(tp, fp, fn)
    jaccardindex = evaluation.jaccardindex(tp, fp, fn)
    jaccarddifference = 1 - jaccardindex

    t.add_row([afile, str(total), str(tp), str(fp), str(tn), str(fn), str(precision), str(sensitivity), str(dicecoeff),
              str(jaccardindex), str(jaccarddifference)])

data = t.get_string()
r = open('segmentation_test_results.txt', 'ab')
r.write(data)
r.close()

r = open('segmentation_test_results.txt', 'a')
r.write("\nCorrect Circle Found Segmentation Results\n")
r.close()

t = PrettyTable(['Version', 'Total', 'True Pos.', 'False Pos.', 'True Neg.', 'False Neg.', 'Precision', 'Sensitivity',
                 'Dice Coeff.', 'Jaccard Ind.', 'Jaccard Dist.'])

for file in files:
    lines = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    f = open(file, 'r')
    for line in f:
        lines += 1
        totals = line.split()
        # totals[0] is image name
        tp += int(totals[1])
        fp += int(totals[2])
        tn += int(totals[3])
        fn += int(totals[4])

    total = tp + fp + tn + fn
    f.close()

    precision = evaluation.precision(tp, fp)
    sensitivity = evaluation.sensitivity(tp, fn)
    fmeasure = evaluation.fmeasure(tp, fp, fn)
    dicecoeff = evaluation.dicecoeff(tp, fp, fn)
    jaccardindex = evaluation.jaccardindex(tp, fp, fn)
    jaccarddifference = 1 - jaccardindex

    t.add_row([file, str(total), str(tp), str(fp), str(tn), str(fn), str(precision), str(sensitivity), str(dicecoeff),
              str(jaccardindex), str(jaccarddifference)])

data = t.get_string()
r = open('segmentation_test_results.txt', 'ab')
r.write(data)
r.close()