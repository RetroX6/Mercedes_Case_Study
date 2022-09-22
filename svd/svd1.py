# get the matrix factors
categories = ['X0','X1', 'X2', 'X3', 'X5','X6', 'X8']
U, S, VT = np.linalg.svd(original_X_train.drop(categories, axis=1),full_matrices=1)
# calculate the aspect ratio β
m = original_X_train.drop(categories, axis=1).shape[1]
n = original_X_train.drop(categories, axis=1).shape[0]
β = m/n

# 0.08422619047619048 near 0.10
# hence let's take w(β) value for β = 0.10 from the above table
ω_β = 1.6089
# find the median singular value
ymed = np.median(S)
# find Hard threshold
cutoff = ω_β * ymed 
print(f"The Hard Threshold for Truncation = {cutoff}")
# get the number of components
r = np.max(np.where(S > cutoff))
print(f"Number of total components to be selected = {r}")
