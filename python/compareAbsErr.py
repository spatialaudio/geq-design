import numpy as np

#import dataset
AbsErr = np.loadtxt("data/test/dataErrorTestAbsolute.csv",delimiter=",")
AbsErrPred = np.loadtxt("data/prediction/predictionErrorTestAbsolute.csv", delimiter=",")
InputData = np.loadtxt("data/test/dataInputTest.csv", delimiter=",")

diff = np.abs(AbsErr - AbsErrPred)

compareErrorAbs = np.zeros((len(AbsErr),1))

for i in range(len(compareErrorAbs)):
    compareErrorAbs[i] = np.mean(diff[i])


countAbs = 0
countHardCases = 0
element = np.zeros(73)
countE = 0

for i in range(len(AbsErrPred)):
    for j in range(AbsErrPred.shape[1]):
        if AbsErrPred[i,j] >= 1:
            element[countE] = i
            countE += 1
            countAbs += 1
            if InputData[i,j] == 12 or InputData[i,j] == -12 : #or InputData[i,j] == 0:
                countHardCases += 1
            break

print("Anzahl Einstellungen mit Fehler größer 1dB:" , countAbs)
print("Anzahl Einstellungen mit Fehler größer 1dB, welche +/- 12 dB Input sind:" , countHardCases)
print("Zeile der harten Einstellungen:", element)