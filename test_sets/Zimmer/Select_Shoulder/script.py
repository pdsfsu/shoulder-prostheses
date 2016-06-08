import os
def GetFiles(dir,f):
    basedir = dir
    subdirs = []
    for fname in os.listdir(dir):
        fileName = os.path.join(basedir, fname)
        if os.path.isfile(fileName):
            if fileName.endswith(".jpg"):
                f.write(fileName.replace('./','')+"\n")
                print fileName.replace('./', '')
        elif os.path.isdir(fileName):
            subdirs.append(fileName)
    for subdir in subdirs:
        GetFiles(subdir,f)
f=open("test.txt",'w')
GetFiles('./', f)
f.close()