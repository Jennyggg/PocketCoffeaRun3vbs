# Convert NANOEDMAODSIM (EDM-format NanoAOD) into flat NANOAODSIM without recomputing anything.
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
opts = VarParsing("analysis")
opts.parseArguments()
process = cms.Process("NANOFLAT")
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(opts.inputFiles))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opts.maxEvents))
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import NANOAODSIMEventContent
process.out = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm=cms.untracked.string("LZMA"), compressionLevel=cms.untracked.int32(9),
    fileName=cms.untracked.string(opts.outputFile),
    outputCommands=NANOAODSIMEventContent.outputCommands,
    dataset=cms.untracked.PSet(dataTier=cms.untracked.string("NANOAODSIM")))
process.end = cms.EndPath(process.out)
