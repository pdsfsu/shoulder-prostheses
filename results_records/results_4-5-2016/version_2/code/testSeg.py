import evaluation

files = ['a.txt', 'ah.txt', 'b.txt', 'bh.txt', 'g.txt', 'gh.txt', 'h.txt', 'n.txt', 'o1.txt', 'o2.txt', 'o3.txt', 'o3_t1-t2.txt', 'o3_t2-t3.txt', 'o3_t3-max.txt']
afiles = ['a_all.txt', 'ah_all.txt', 'b_all.txt', 'bh_all.txt', 'g_all.txt', 'gh_all.txt', 'h_all.txt', 'n_all.txt', 'o1_all.txt', 'o2_all.txt', 'o3_all.txt', 'o3_t1-t2_all.txt', 'o3_t2-t3_all.txt', 'o3_t3-max_all.txt']

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
    f.close()

    precision = evaluation.precision(tp, fp)
    sensitivity = evaluation.sensitivity(tp, fn)
    fmeasure = evaluation.fmeasure(tp, fp, fn)
    dicecoeff = evaluation.dicecoeff(tp, fp, fn)
    jaccardindex = evaluation.jaccardindex(tp, fp, fn)

    r = open('circle_found_results.txt', 'a')
    r.write('File name: ' + file + '\n')
    r.write('Lines: ' + str(lines) + '\n')
    r.write('True Positives: ' + str(tp) + '\n')
    r.write('False Positives: ' + str(fp) + '\n')
    r.write('True Negatives: ' + str(tn) + '\n')
    r.write('False Negatives: ' + str(fn) + '\n')
    r.write('Precision: ' + str(precision) + '\n')
    r.write('Sensitivity: ' + str(sensitivity) + '\n')
    r.write('F-Measure: ' + str(fmeasure) + '\n')
    r.write('Dice Coefficent: ' + str(dicecoeff) + '\n')
    r.write('Jaccard Index: ' + str(jaccardindex) + '\n' + '\n')
    r.close()

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
    af.close()

    precision = evaluation.precision(tp, fp)
    sensitivity = evaluation.sensitivity(tp, fn)
    fmeasure = evaluation.fmeasure(tp, fp, fn)
    dicecoeff = evaluation.dicecoeff(tp, fp, fn)
    jaccardindex = evaluation.jaccardindex(tp, fp, fn)

    r = open('all_results.txt', 'a')
    r.write('File name: ' + afile + '\n')
    r.write('Lines: ' + str(lines) + '\n')
    r.write('True Positives: ' + str(tp) + '\n')
    r.write('False Positives: ' + str(fp) + '\n')
    r.write('True Negatives: ' + str(tn) + '\n')
    r.write('False Negatives: ' + str(fn) + '\n')
    r.write('Precision: ' + str(precision) + '\n')
    r.write('Sensitivity: ' + str(sensitivity) + '\n')
    r.write('F-Measure: ' + str(fmeasure) + '\n')
    r.write('Dice Coefficent: ' + str(dicecoeff) + '\n')
    r.write('Jaccard Index: ' + str(jaccardindex) + '\n' + '\n')
    r.close()
