from sklearn.decomposition import TruncatedSVD

tsvd = TruncatedSVD(n_components= r, random_state=420)
tsvd_train = tsvd.fit_transform(original_X_train.drop(["X0","X1","X2","X3","X5","X6","X8"], axis=1))
tsvd_cv = tsvd.transform(original_X_cv.drop(["X0","X1","X2","X3","X5","X6","X8"], axis=1))
tsvd_test = tsvd.transform(original_X_test.drop(["X0","X1","X2","X3","X5","X6","X8"], axis=1))

print(tsvd_train.shape)
print(tsvd_cv.shape)
print(tsvd_test.shape)
