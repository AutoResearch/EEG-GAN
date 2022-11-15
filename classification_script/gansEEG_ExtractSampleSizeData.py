import numpy as np
import random as rnd

EEGData = np.genfromtxt('ganTrialERP_len100.csv', delimiter=',', skip_header=1)
testParticipants = np.sort(rnd.sample(range(1,int(np.unique(EEGData[:,0])[-1])+1), 400))

testEEGData = []
for participant in testParticipants:
    testEEGData.extend(EEGData[EEGData[:,0]==participant,:])
    EEGData = np.delete(EEGData, np.where(EEGData[:,0]==participant), axis=0)  

saveFilename = 'Reduced/gansTrialERP_len100_TestSS400.csv'
np.savetxt(saveFilename, testEEGData, delimiter=",")
    
for sampleSize in np.arange(5,101,5):
    sampleParticipants = rnd.sample(range(1,int(np.unique(EEGData[:,0])[-1])+1), sampleSize)
    
    reducedData = []
    for participant in sampleParticipants:
        reducedData.extend(EEGData[EEGData[:,0]==participant,:])

    saveFilename = 'Reduced/gansTrialERP_len100_SS' + str(sampleSize).zfill(3) + '.csv'
    np.savetxt(saveFilename, reducedData, delimiter=",")