from Data.Datasets.globalSplit import globalSplit
from Data.Datasets.oneSubjectOutCrossValidation import oneSubjectOutCrossValidation

# x1,x2,x3,x4,x5,x6 = globalSplit('../Raw/x2DnoBaselineStd.csv', '../Raw/y2D.csv')
# x1,x2,x3,x4,x5,x6 = globalSplit('../Raw/x2DnoBaselineUnStd.csv', '../Raw/y2D.csv')
# x1,x2,x3,x4,x5,x6 = globalSplit('../Raw/x2DreducedUnStd.csv', '../Raw/y2D.csv')
# x1,x2,x3,x4,x5,x6 = globalSplit('../Raw/x2DreducedStd.csv', '../Raw/y2D.csv')
# x1,x2,x3,x4,x5,x6 = globalSplit('../Raw/x2DStd.csv', '../Raw/y2D.csv')
# x1,x2,x3,x4,x5,x6 = globalSplit('../Raw/x2DUnStd.csv', '../Raw/y2D.csv')
# x1,x2 = oneSubjectOutCrossValidation('globalSplit/unstd/Xtr_Unstd.npy', 'globalSplit/unstd/Ttr_Unstd.npy')
# x1,x2 = oneSubjectOutCrossValidation('globalSplit/std/Xtr_Std.npy', 'globalSplit/std/Ttr_Std.npy')
# x1,x2 = oneSubjectOutCrossValidation('globalSplit/reducedStd/Xtr_reducedStd.npy', 'globalSplit/reducedStd/Ttr_reducedStd.npy')
# x1,x2 = oneSubjectOutCrossValidation('globalSplit/reducedUnstd/Xtr_reducedUnstd.npy', 'globalSplit/reducedUnstd/Ttr_reducedUnstd.npy')
# x1,x2 = oneSubjectOutCrossValidation('globalSplit/noBaselineUnstd/Xtr_noBaselineUnstd.npy', 'globalSplit/noBaselineUnstd/Ttr_noBaselineUnstd.npy')
x1,x2 = oneSubjectOutCrossValidation('globalSplit/noBaselineStd/Xtr_noBaselineStd.npy', 'globalSplit/noBaselineStd/Ttr_noBaselineStd.npy')