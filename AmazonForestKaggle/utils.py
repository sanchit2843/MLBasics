import numpy as np
def get_pred(y_pred):
    l = len(y_pred)
    y_pred = y_pred[0:l-1]
    y_pred = np.asarray(y_pred)
    for i in range(len(y_pred)):
        for j in range(16):
            for k in range(17):
                try:
                    if(y_pred[i][j][k]>=0.5):
                        y_pred[i][j][k] = 1
                    else:
                        y_pred[i][j][k] = 0
                except:
                    print(y_pred.shape)
    return y_pred
def get_fscore(y_true,y_pred):
    leng = len(y_true)
    y_true = y_true[0:leng-1]
    y_true = np.asarray(y_true)
    leng = len(y_true)
    siz = leng*16
    y_true = np.reshape(y_true,(siz,17))
    y_pred = np.reshape(y_pred,(siz,17))

    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    return f1
def error_plot(loss):
    plt.figure(figsize=(10,5))
    plt.plot(loss)
    plt.title("Training loss plot")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.show()
def f1score(f1score):
    plt.figure(figsize=(10,5))
    plt.plot(f1score)
    plt.title("Training f1score plot")
    plt.xlabel("epochs")
    plt.ylabel("f1score")
    plt.show()
