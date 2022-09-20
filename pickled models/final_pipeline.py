import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import r2_score
import joblib
import warnings
warnings.filterwarnings("ignore") 

def clean_categorical(data):
    
    """
    This function takes the dataframe as input and
    encodes the categorical features.
    """
    # create empty lists for collecting feature names
    cat_features = ['X0','X1','X2','X3','X4','X5','X6','X8']
    
    Binary_features = ['X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 
                       'X24', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39',
                       'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 
                       'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 
                       'X70', 'X71', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85',
                       'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93', 'X94', 'X95', 'X96', 'X97', 'X98', 'X99', 'X100',
                       'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X107', 'X108', 'X109', 'X110', 'X111', 'X112', 'X113',
                       'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120', 'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 
                       'X128', 'X129', 'X130', 'X131', 'X132', 'X133', 'X134', 'X135', 'X136', 'X137', 'X138', 'X139', 'X140',
                       'X141', 'X142', 'X143', 'X144', 'X145', 'X146', 'X147', 'X148', 'X150', 'X151', 'X152', 'X153', 'X154', 
                       'X155', 'X156', 'X157', 'X158', 'X159', 'X160', 'X161', 'X162', 'X163', 'X164', 'X165', 'X166', 'X167', 
                       'X168', 'X169', 'X170', 'X171', 'X172', 'X173', 'X174', 'X175', 'X176', 'X177', 'X178', 'X179', 'X180', 
                       'X181', 'X182', 'X183', 'X184', 'X185', 'X186', 'X187', 'X189', 'X190', 'X191', 'X192', 'X194', 'X195', 
                       'X196', 'X197', 'X198', 'X199', 'X200', 'X201', 'X202', 'X203', 'X204', 'X205', 'X206', 'X207', 'X208', 
                       'X209', 'X210', 'X211', 'X212', 'X213', 'X214', 'X215', 'X216', 'X217', 'X218', 'X219', 'X220', 'X221',
                       'X222', 'X223', 'X224', 'X225', 'X226', 'X227', 'X228', 'X229', 'X230', 'X231', 'X232', 'X233', 'X234', 
                       'X235', 'X236', 'X237', 'X238', 'X239', 'X240', 'X241', 'X242', 'X243', 'X244', 'X245', 'X246', 'X247', 
                       'X248', 'X249', 'X250', 'X251', 'X252', 'X253', 'X254', 'X255', 'X256', 'X257', 'X258', 'X259', 'X260', 
                       'X261', 'X262', 'X263', 'X264', 'X265', 'X266', 'X267', 'X268', 'X269', 'X270', 'X271', 'X272', 'X273', 
                       'X274', 'X275', 'X276', 'X277', 'X278', 'X279', 'X280', 'X281', 'X282', 'X283', 'X284', 'X285', 'X286',
                       'X287', 'X288', 'X289', 'X290', 'X291', 'X292', 'X293', 'X294', 'X295', 'X296', 'X297', 'X298', 'X299',
                       'X300', 'X301', 'X302', 'X304', 'X305', 'X306', 'X307', 'X308', 'X309', 'X310', 'X311', 'X312', 'X313', 
                       'X314', 'X315', 'X316', 'X317', 'X318', 'X319', 'X320', 'X321', 'X322', 'X323', 'X324', 'X325', 'X326', 
                       'X327', 'X328', 'X329', 'X330', 'X331', 'X332', 'X333', 'X334', 'X335', 'X336', 'X337', 'X338', 'X339', 
                       'X340', 'X341', 'X342', 'X343', 'X344', 'X345', 'X346', 'X347', 'X348', 'X349', 'X350', 'X351', 'X352',
                       'X353', 'X354', 'X355', 'X356', 'X357', 'X358', 'X359', 'X360', 'X361', 'X362', 'X363', 'X364', 'X365',
                       'X366', 'X367', 'X368', 'X369', 'X370', 'X371', 'X372', 'X373', 'X374', 'X375', 'X376', 'X377', 'X378', 
                       'X379', 'X380', 'X382', 'X383', 'X384', 'X385']
    
    
    # create categorical feature dataframe
    cat_df = data[cat_features]
    oe = OrdinalEncoder()
    enc_cat_df = oe.fit_transform(cat_df)
    enc_df = pd.DataFrame(enc_cat_df,columns=cat_features)
    enc_df.insert(0, 'ID', data['ID'].values)
    binary_df = data.drop(['X0', 'X1', 'X2', 'X3', 'X4','X5', 'X6', 'X8'],axis=1)
    for col in data.columns:
        if col=='y':
            binary_df = binary_df.drop(['y'],axis=1)
            break
            
    final_df = pd.merge(enc_df,binary_df,on='ID',how='left')

    return final_df.drop(['ID'],axis=1)

################################################################################################################################

# creating a final prediction function

def final_fun_1(X):
    
    # preprocess the data
    X_processed = clean_categorical(X)
    #load the saved model 1
    model1 = joblib.load("final_best_model1_xgb.pkl")
    # predict the targets
    y_pred1 = model1.predict(X_processed)
    dtest = xgb.DMatrix(X_processed)
    #load the saved model 2
    model2 = joblib.load("final_best_model2_xgb.pkl")
    # predict the targets
    y_pred2 = model2.predict(dtest)
    # Average the test data preditions of both models
    avg_pred = (y_pred1 + y_pred2)/2
    
    model3 = joblib.load("final_best_model3_stacked.pkl")
    y_pred3 = model3.predict(X_processed)
    
    #final prediction: taking averaged models of 1 and 2 as output and aveaging the output with stacked model output
    final_prediction = (avg_pred + y_pred3)/2
    
    return final_prediction

################################################################################################################################


def final_fun_2(X,y):
    # preprocess the data
    X_processed = clean_categorical(X)
    # load the saved model 1
    model1 = joblib.load("final_best_model1_xgb.pkl")
    # predict the targets
    y_pred1 = model1.predict(X_processed)
    dtest = xgb.DMatrix(X_processed)
    #load the saved model 2
    model2 = joblib.load("final_best_model2_xgb.pkl")
    # predict the targets
    y_pred2 = model2.predict(dtest)
    # Average the test data preditions of both models
    avg_pred = (y_pred1 + y_pred2)/2
    #load the saved model 3
    model3 = joblib.load("final_best_model3_stacked.pkl")
    y_pred3 = model3.predict(X_processed)
    
    #final prediction: taking averaged models of 1 and 2 as output and aveaging the output with stacked model output
    final_prediction = (avg_pred + y_pred3)/2
    
    return r2_score(y,final_prediction)
