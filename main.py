from utilities.parserUtils import *
from utilities.customUtils import *
from utilities.aestheticUtils import *
from dataTools.processDataset import *
from dataTools.patchExtractor import *
from mainModule.MIDRAN import *

if __name__ == "__main__":

    # Parsing Options
    options = mainParser(sys.argv[1:])
    if len(sys.argv) == 1:
        customPrint("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()
    
    # Reading Model Configuration
    if options.conf:
        configCreator()

    # Loading Configuration
    config = configReader()
  
    # Taking action as per received options
    if options.epoch:
        config=updateConfig(entity='epoch', value=options.epoch)
    if options.batch:
        config=updateConfig(entity='batchSize', value=options.batch)
    if options.manualUpdate:
        config=manualUpdateEntity()
    if options.modelSummary:
        MIDRAN(config).modelSummary()
    if options.train:
        MIDRAN(config).modelTraining(dataSamples=options.dataSamples)
    if options.retrain:
        MIDRAN(config).modelTraining(resumeTraning=True, dataSamples=options.dataSamples) 
    if options.inference:
        noiseSigmaSet = None
        if options.noiseSigma:
            noiseSigmaSet = options.noiseSigma.split(',')
            noiseSigmaSet = list(map(int, noiseSigmaSet))
        MIDRAN(config).modelInference(testImagesPath=options.sourceDir, outputDir=options.resultDir, noiseSet=noiseSigmaSet)
    if options.overFitTest:
        MIDRAN(config).modelTraining(overFitTest=True)
    
        
        
        
            


