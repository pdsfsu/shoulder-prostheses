import evaluation

files = ['a.txt', 'ah.txt', 'b.txt', 'bh.txt', 'g.txt', 'gh.txt', 'h.txt', 'n.txt', 'o1.txt', 'o2.txt', 'o3.txt']

for file in files:
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    f = open(file, 'r')
    for line in f:
       totals = line.split()
       tp += totals[0]
       fp += totals[1]
       tn += totals[2]
       fn += totals[3]
    f.close()

    precision = evaluation.precision(tp, fp)
    sensitivity = evaluation.sensitivity(tp, fn)
    fmeasure = evaluation.fmeasure(tp, fp, fn)
    dicecoeff = evaluation.dicecoeff(tp, fp, fn)
    jaccardindex = evaluation.jaccardindex(tp, fp, fn)

    r = open('all_results.txt', 'a')
    r.write('File name: ' + file + '\n')
    r.write('True Positives: ' + tp + '\n')
    r.write('False Positives: ' + fp + '\n')
    r.write('True Negatives: ' + tn + '\n')
    r.write('False Negatives: ' + fp + '\n')
    r.write('Precision: ' + precision + '\n')
    r.write('Sensitivity: ' + sensitivity + '\n')
    r.write('F-Measure: ' + fmeasure + '\n')
    r.write('Dice Coefficent: ' + dicecoeff + '\n')
    r.write('Jaccard Index: ' + jaccardindex + '\n' + '\n')
    r.close()

for file in files:
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    f = open(file, 'r')
    for line in f:
       totals = line.split()
       if int(totals[0]) > 0:
           tp += totals[0]
           fp += totals[1]
           tn += totals[2]
           fn += totals[3]
    f.close()

    precision = evaluation.precision(tp, fp)
    sensitivity = evaluation.sensitivity(tp, fn)
    fmeasure = evaluation.fmeasure(tp, fp, fn)
    dicecoeff = evaluation.dicecoeff(tp, fp, fn)
    jaccardindex = evaluation.jaccardindex(tp, fp, fn)

    r = open('circle_found_results.txt', 'a')
    r.write('File name: ' + file + '\n')
    r.write('True Positives: ' + tp + '\n')
    r.write('False Positives: ' + fp + '\n')
    r.write('True Negatives: ' + tn + '\n')
    r.write('False Negatives: ' + fp + '\n')
    r.write('Precision: ' + precision + '\n')
    r.write('Sensitivity: ' + sensitivity + '\n')
    r.write('F-Measure: ' + fmeasure + '\n')
    r.write('Dice Coefficent: ' + dicecoeff + '\n')
    r.write('Jaccard Index: ' + jaccardindex + '\n' + '\n')
    r.close()