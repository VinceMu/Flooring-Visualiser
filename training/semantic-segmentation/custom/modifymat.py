from scipy.io import loadmat, savemat
annots = loadmat('../data/color150.mat')
annots['colors'] = annots['colors'][3:4]
print(annots['colors'])
savemat("../data/color1.mat", annots)
