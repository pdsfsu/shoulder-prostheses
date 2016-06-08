from prettytable import PrettyTable
import evaluation
import sys

arg_list = sys.argv

if len(arg_list) < 6:
    sys.exit("Missing input parameters: version true positive, false positive, true negative, false negative")

version = arg_list[1]
tp = int(arg_list[2])
fp = int(arg_list[3])
tn = int(arg_list[4])
fn = int(arg_list[5])
total = tp + fp + tn + fn
precision = evaluation.precision(tp, fp)
sensitivity = evaluation.sensitivity(tp, fn)
fmeasure = evaluation.fmeasure(tp, fp, fn)
dicecoeff = evaluation.dicecoeff(tp, fp, fn)
jaccardindex = evaluation.jaccardindex(tp, fp, fn)
jaccarddifference = 1 - jaccardindex

# example table
print "Circle Detection\n"
t = PrettyTable(['Version', 'Total', 'True Pos.', 'Fasle Pos.', 'True Neg.', 'False Neg.', 'Precision', 'Sensitivity',
                 'Dice Coeff.', 'Jaccard Ind.', 'Jaccard Dist.'])
t.add_row(['3/15', str(53), str(24), str(26), str(0), str(3), str(.48), str(.89), str(.62), str(.45), str(.55)])
t.add_row(['4/5', str(53), str(46), str(7), str(0), str(0), str(.87), str(1), str(.93), str(.87), str(.13)])
t.add_row([version, str(total), str(tp), str(fp), str(tn), str(fn), str(precision), str(sensitivity), str(dicecoeff),
              str(jaccardindex), str(jaccarddifference)])
print t

data = t.get_string()
r = open('circle_test_results.txt', 'ab')
r.write(data)
r.close()

