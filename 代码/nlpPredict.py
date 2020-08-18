#!-*- coding:utf-8 -*-
from _operator import index
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xlrd import sheet
from math import isnan
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import datetime as dt
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import xlsxwriter
from time import sleep
def iter_files(root_dir):
    #遍历根目录
    all_pic_path = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            file_name = os.path.join(root,file)
            if file[0] == '.':
                continue
            all_pic_path.append(file_name)
    return all_pic_path
    # all_pic_path = []
    # for root, dirs, files in os.walk(root_dir):
    #     files = [f for f in files if not f[0] == '.']
    #     dirs[:] = [d for d in dirs if not d[0] == '.']
    #     # use files and dirs
    #     for file_name in files:
    #         file_name = os.path.join(root, file_name)
    #         all_pic_path.append(file_name)
    #     return all_pic_path

def sentimentScore(Tweet):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for sentence in Tweet:
        vs = analyzer.polarity_scores(sentence)
        #print("Vader score: " + str(vs))
        results.append(vs)
    return results

def standardscaler(score):
    scaler = StandardScaler().fit(score)
    scaled_data = scaler.transform(score)
    return scaled_data

#自己实现knn
def classify0(inx,dataset,labels,k):
    dist = np.sum((inx - dataset) ** 2, axis=1) ** 0.5
    k_labels = [labels[index] for index in dist.argsort()[0: k]]
    # 出现次数最多的标签即为最终类别
    k_ner = pd.DataFrame(k_labels, columns=['label'])
    k_ner = k_ner['label'].value_counts().reset_index()
    # print(k_ner)
    label = k_ner.loc[0, 'index']
    return label

#自己实现朴素贝叶斯
def classify(trainData, labels, features):

    # 求labels中每个label的先验概率
    labels = list(labels)  # 转换为list类型
    labelset = set(labels)
    P_y = {}  # 存入label的概率
    for label in labelset:
        P_y[label] = labels.count(label) / float(len(labels))  # p = count(y) / count(Y)
        # print(label,P_y[label])

        # 求label与feature同时发生的概率
    P_xy = {}
    for y in P_y.keys():
        y_index = [i for i, label in enumerate(labels) if label == y]  # labels中出现y值的所有数值的下标索引
        for j in range(len(features)):  # features[0] 在trainData[:,0]中出现的值的所有下标索引
            x_index = [i for i, feature in enumerate(trainData[:, j]) if feature == features[j]]
            xy_count = len(set(x_index) & set(y_index))  # set(x_index)&set(y_index)列出两个表相同的元素
            pkey = str(features[j]) + '*' + str(y)
            P_xy[pkey] = xy_count / float(len(labels))
            # print(pkey,P_xy[pkey])

        # 求条件概率
    P = {}
    for y in P_y.keys():
        for x in features:
            pkey = str(x) + '|' + str(y)
            P[pkey] = P_xy[str(x) + '*' + str(y)] / float(P_y[y])  # P[X1/Y] = P[X1Y]/P[Y]
            # print(pkey,P[pkey])

        # 求[2,'S']所属类别
    F = {}  # [2,'S']属于各个类别的概率
    for y in P_y:
        F[y] = P_y[y]
        for x in features:
            F[y] = F[y] * P[
                str(x) + '|' + str(y)]  # P[y/X] = P[X/y]*P[y]/P[X]，分母相等，比较分子即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]
            # print(str(x),str(y),F[y])

    features_label = max(F, key=F.get)  # 概率最大值对应的类别
    #print(features_label)
    return features_label


#自己实现逻辑回归
# 计算sigmoid函数
def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))

### $梯度上升的公式： w = w +af(x) $
# 逻辑回归训练模型
# 输入为训练集中的数值和属性，以及算法的选择
# 返回各个特征的权重值代表回归系数，并且计算训练的时间
def trainLogRegres(train_x, train_y, opts):
    # startTime = time.time()  # 计算时间
    # train_x = np.mat(train_x)

    numSamples, numFeatures = np.shape(train_x)  # 获取数据集的维度

    alpha = opts['alpha']  # 获取学习的步长
    maxIter = opts['maxIter']  # 选择迭代的次数
    weights = np.ones((numFeatures, 1))  # 每个特征一个权值，初始化数值为1

    # 通过梯度上升算法进行优化
    for k in range(maxIter):
        # 迭代，接下来选择计算w的方式1.
        if opts['optimizeType'] == 'gradUp':  # 梯度上升优化算法。
            output = sigmoid(train_x * weights)  # 矩阵运算，计算量大，计算复杂度高，每次更新时都要遍历。
            error = train_y - output  # 错误率
            weights += alpha * train_x.transpose() * error

        elif opts['optimizeType'] == 'stocGradUp':  # 随机梯度上升，增量式更新，批处理，向量运算（但存在较大误差）
            for i in range(numSamples):  # 每一行
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights += alpha * train_x[i, :].transpose() * error  #

        elif opts['optimizeType'] == 'smoothStocGradUp':  # 平滑随机梯度下降
            # 随机选择样本以优化以减少周期波动
            dataIndex = list(range(numSamples))  # 索引
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01  # a每次迭代时都需要调整，以缓解数据波动。
                randIndex = int(np.random.uniform(0, len(dataIndex)))  # 随机生成引索
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex] - output
                weights += alpha * train_x[randIndex, :].transpose() * error
                del (dataIndex[randIndex])  # 交互后删除优化的样本

        else:
            raise NameError('NOT FOUND！')
    # print('完成本次训练需要 %fs!' % (time.time() - startTime))

    return weights  # 返回权重向量。
# 测试函数，输入为训练得到的回归系数以及测试集的数值和属性
# def testLogRegres(weights, test_x, test_y):
#     numSamples, numFeatures = np.shape(test_x)  # 获得测试数据的大小
#     matchCount = 0  # 记录匹配率
#     for i in range(numSamples):
#         predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5  # 与0.5做判断
#         # print(predict)
#         if predict == bool(test_y[i]):
#             matchCount += 1
#     accuracy = float(matchCount) / numSamples
#     return accuracy




if __name__ == "__main__":
    # 文件路径
    # 读取sheet的名字
    # 训练98支股票，依次得到训练准确率
    all_stock_path = iter_files('stock')
    # all_stock_path.remove()
    sheetName = 'Stream'
    # cols = ['Tweet content']
#######推特数据########
    for i in range(0, len(all_stock_path), 2):
        if len(all_stock_path[i]) < len(all_stock_path[i+1]):
            stock_tweet = all_stock_path[i]
            stock_price = all_stock_path[i + 1]
        else:
            stock_tweet = all_stock_path[i + 1]
            stock_price = all_stock_path[i]
        stock_name = stock_tweet[stock_tweet.rfind('/')+1:stock_tweet.rfind('.')]
        stock_path = stock_tweet[:stock_tweet.rfind('/') + 1]
        # stock_tweet = 'stock/AAL/AAL.xlsx'
        # stock_price = 'stock/AAL/AAL_stock.xlsx'
        # stock_name = 'AAL'
        # stock_path = 'stock/AAL/'
        print('现在跑的文件:'+stock_tweet)
        # print(stock_price)
        # print(stock_name)
        # print(stock_path)
        # continue
        # 读取tweet数据
        df = pd.read_excel(stock_tweet, sheet_name=sheetName)
        # print(df)
        file = df['Tweet content']
        # file = pd.read_excel("AAL.xlsx", sheet_name=sheetName,usecols = cols)

        # 计算每条tweet的四种情感得分
        Tweets = []
        for i in range(len(file)):
            Tweets.append(file.loc[i])
        df_results = pd.DataFrame(sentimentScore(Tweets))
        # print(df_results)

        # 将得分作为新的数据列与 Stream 数据合并
        df['neg'] = df_results['neg']
        df['neu'] = df_results['neu']
        df['pos'] = df_results['pos']
        df['compound'] = df_results['compound']
        # print("将得分作为新的数据列与 Stream 数据合并:")
        # print(df)
        df.to_excel(stock_tweet, sheet_name=sheetName, index=None)

        # 选取任意连续 60 天的所有数据
        # 将 Date 列转换为日期格式
        # df_1 = df.copy()
        df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'])
        df_1 = df.copy()
        # print(df_1)
        # print("将 Date 列转换为日期格式:")
        # print(df_1['Date'])
        # 将新的日期设置为index
        df.set_index("Date", inplace=True)

        # 挑选出有价值的属性['Hour', 'Tweet content', 'Favs', 'RTs', 'Followers', 'Following', 'Is a RT', 'Hashtags', 'Symbols', 'compound', 'neg', 'neu', 'pos']；
        result = pd.DataFrame(columns=(
        'Date', 'Hour', 'Tweet content', 'Favs', 'RTs', 'Followers', 'Following', 'Is a RT', 'Hashtags', 'Symbols',
        'compound', 'neg', 'neu', 'pos'))

        result.loc[:, 'Date'] = pd.to_datetime(df_1.loc[:, 'Date'])
        result.loc[:, 'Hour'] = df_1.loc[:, 'Hour']
        result.loc[:, 'Tweet content'] = df_1.loc[:, 'Tweet content']
        result.loc[:, 'Favs'] = df_1.loc[:, 'Favs']
        result.loc[:, 'RTs'] = df_1.loc[:, 'RTs']
        result.loc[:, 'Followers'] = df_1.loc[:, 'Followers']
        result.loc[:, 'Following'] = df_1.loc[:, 'Following']
        result.loc[:, 'Is a RT'] = df_1.loc[:, 'Is a RT']
        result.loc[:, 'Hashtags'] = df_1.loc[:, 'Hashtags']
        result.loc[:, 'Symbols'] = df_1.loc[:, 'Symbols']
        result.loc[:, 'compound'] = df_1.loc[:, 'compound']
        result.loc[:, 'neg'] = df_1.loc[:, 'neg']
        result.loc[:, 'neu'] = df_1.loc[:, 'neu']
        result.loc[:, 'pos'] = df_1.loc[:, 'pos']
        # print(result)
        # 去掉 compound 的得分为 0（即中立）的数据
        # 去掉 Followers 为空的数据；
        for i in range(len(result)):
            if result.loc[i, 'compound'] == 0 or isnan(result.loc[i, 'Followers']):
                result.drop(i, axis=0, inplace=True)
        result_index = result.reset_index()

        # 增加新的一列 Compound_multiplied，由 compound 和 Followers 相乘而来
        result_index['Compound_multiplied'] = result_index['compound'] * result_index['Followers']

        # 将Compound_multiplied标准化，作为新的一列Compound_multiplied_scaled
        score = pd.DataFrame(result_index['Compound_multiplied']).values.astype(float)
        result_index['Compound_multiplied_scaled'] = pd.DataFrame(standardscaler(score))

        # 将新的日期设置为index
        result_index.set_index("Date", inplace=True)
        time_list = pd.date_range(start='20160310', periods=97, freq='1D')
        # print(time_list)
        # time_list = [end_time - datetime.timedelta(days=i) for i in range(60)]

        result_1 = pd.DataFrame(
            columns=('Date', 'Favs', 'RTs', 'Followers', 'Following', 'Is a RT', 'compound', 'neg', 'neu', 'pos',
                     'Compound_multiplied', 'Compound_multiplied_scaled'))
        k = 0
        for time_ele in time_list:
            result_1.loc[k, 'Date'] = time_ele
            time_ele = pd.datetime.strftime(time_ele, "%Y-%m-%d")
            # result_1.loc[k,'Date'] = time_ele
            # 计算均值
            result_ave = result_index[time_ele].mean()
            result_1.loc[k, 1:] = result_ave
            k += 1
        for i in range(10,16):
            result_1.drop(i, axis=0, inplace=True)
        result_2 = result_1.reset_index()


    #######股票#########

        #从网上爬取股票数据
        start = dt.datetime(2016, 3, 10)
        end = dt.datetime(2016, 6, 15)  # dt.datetime.now()

        df = pd.read_excel(stock_price)

        # 每日的股价涨跌情况（百分制）
        df.loc[0,'Change'] = ((df.loc[0,'Close'] - df.loc[0,'Open']) / df.loc[0,'Open']) * 100
        for i in range(1,len(df)):
            df.loc[i,'Change'] = ((df.loc[i,'Close'] - df.loc[i-1,'Close'])/df.loc[i-1,'Adj Close']) * 100
        #（t日的收盘-（t-1）日的收盘）/t-1日调整后的收盘价 * 100

        #合并两个表
        # print(df)
        result_2 = result_2.set_index('Date')
        result_2.to_excel(stock_path + "result.xlsx")
        result_2 = pd.merge(result_2, df, on='Date')
        result_2.to_excel(stock_path + "new1.xlsx")

        #记录下一天的涨跌情况
        for i in range(len(result_2)-1):
            result_2.loc[i,'Tomorrow'] = result_2.loc[i+1,'Change']
            if result_2.loc[i,'Tomorrow'] > 0:
                result_2.loc[i,'Buy/Sell'] = 1
            else:
                result_2.loc[i, 'Buy/Sell'] = -1

        #数据填充
        result_2.to_excel(stock_path+"1.xlsx",index = None)
        result_2['Tomorrow'] = pd.to_numeric(result_2['Tomorrow'])
        result_2['Buy/Sell'].fillna(1,inplace=True)

        result_2.to_excel(stock_path+"$" + "a" + stock_name + ".xlsx", index=None)
        result_3 = pd.read_excel(stock_path+"$" + "a"+stock_name + ".xlsx")
        result_3.fillna(result_3.mean(),inplace=True)
        result_3.to_excel(stock_path+"$" + stock_name + ".xlsx", index=None)

        # result_3.fillna(0, inplace=True)

        # result_2.loc[90]['Compound_multiplied_scaled']=-0.07
        # result_2['Buy/Sell'] = pd.to_numeric(result_2['Buy/Sell'])

        # 拆分测试集、训练集
        X_train, X_test, y_train, y_test = train_test_split(result_3['Compound_multiplied_scaled']
                                                            , result_3['Buy/Sell'],test_size=0.2)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train.reshape(-1,1))
        X_test = ss.transform(X_test.reshape(-1,1))

        #knn（sklearn）
        sklearn_knn_clf = KNeighborsClassifier(n_neighbors=6)
        sklearn_knn_clf.fit(X_train, y_train)
        accu = sklearn_knn_clf.score(X_test.reshape(-1,1),y_test.reshape(-1,1))
        print("knn 对X_train,Y_train的 R值(准确率):", accu)
        y_predict = sklearn_knn_clf.predict(X_test.reshape(-1,1))
        # print(y_predict)
        listKNNpre = y_predict.tolist()
        #print('listKNNpre:')
        #print(listKNNpre)

        #截取要进行预测的部分
        n = len(result_3)
        nPredict = len(listKNNpre)
        preDF = result_3.head(n-nPredict)
        preDF.to_excel(stock_path+stock_name+"_train.xlsx")
        preDF = result_3.tail(nPredict)
        preDF.to_excel(stock_path+stock_name+"_prediction.xlsx")



        #添加一列buy-hold收益
        adjC =preDF['Adj Close'].tolist()
        base = adjC[0]
        preDF['buy-hold'] =preDF['Adj Close']/base
        BuyHold = preDF['buy-hold'].tolist()


        open = preDF['Open'].tolist()
        close = preDF['Close'].tolist()

        re = []
        for i in range(nPredict):
            if listKNNpre[i] == 1:
                tem = close[i]/ open[i]
                re.append(tem)
            else:
                re.append(1)
        #print(re)
        for i in range(1,nPredict):
            re[i] = re[i-1] * re[i]


        x = np.linspace(0, nPredict, nPredict)
        plt.figure()  # 使用plt.figure定义一个图像窗口.
        plt.plot(x, BuyHold,label='buy-hold')  # 使用plt.plot画(x ,y2)曲线.
        # 使用plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色;
        # 曲线的宽度(linewidth)为1.0；
        # 曲线的类型(linestyle)为虚线.
        plt.plot(x, re, color='red',label='knn')

        plt.xlabel('day')
        plt.ylabel('re')
        # set line syles

        # legend将要显示的信息来自于上面代码中的 label. 所以我们只需要简单写下一下代码, plt 就能自动的为我们添加图例.
        plt.legend(loc='upper right')
        plt.savefig(stock_path+'knn.png')
        #plt.show()

        #preDF['knnchange'] = re

        #print(BuyHold)
        #preDF.to_excel("preDF.xlsx")


        #LogisticRegression
        cls = LogisticRegression()
        cls.fit(X_train, y_train)

        rls = cls.score(X_test, y_test)

        print("LogisticRegression 对X_train,Y_train的 R值(准确率):", rls)

        # print(X_test)
        Y_predict = cls.predict(X_test)
        listLogisticRegpre = Y_predict.tolist()
        #print("LogisticRegression预测结果：")
        #print(listLogisticRegpre)

        re = []
        for i in range(nPredict):
            if listLogisticRegpre[i] == 1:
                tem = close[i]/ open[i]
                re.append(tem)
            else:
                re.append(1)
        #print(re)
        for i in range(1,nPredict):
            re[i] = re[i-1] * re[i]


        x = np.linspace(0, nPredict, nPredict)

        plt.figure()  # 使用plt.figure定义一个图像窗口.
        plt.plot(x, BuyHold,label='buy-hold')  # 使用plt.plot画(x ,y2)曲线.
        # 使用plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色;

        plt.plot(x, re, color='red',label='LogisticRegression')

        plt.xlabel('day')
        plt.ylabel('re')
        # set line syles

        # legend将要显示的信息来自于上面代码中的 label. 所以我们只需要简单写下一下代码, plt 就能自动的为我们添加图例.
        plt.legend(loc='upper right')
        plt.savefig(stock_path+'LogisticRegression.png')
        #plt.show()

        # SVM

        svm = SVC()
        svm.fit(X_train, y_train)  # 训练
        rsvm = svm.score(X_test, y_test)
        print("svm 对X_test,Y_test的 R值(准确率):", rsvm)
        result = svm.predict(X_test)
        # print("svm预测结果：")
        # print(result)
        listSVMpre = result.tolist()
        #print("LogisticRegression预测结果：")
        #print(listLogisticRegpre)

        re = []
        for i in range(nPredict):
            if listSVMpre[i] == 1:
                tem = close[i]/ open[i]
                re.append(tem)
            else:
                re.append(1)
        #print(re)
        for i in range(1,nPredict):
            re[i] = re[i-1] * re[i]


        x = np.linspace(0, nPredict, nPredict)
        plt.figure()  # 使用plt.figure定义一个图像窗口.
        plt.plot(x, BuyHold,label='buy-hold')  # 使用plt.plot画(x ,y2)曲线.
        # 使用plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色;
        # 曲线的宽度(linewidth)为1.0；
        # 曲线的类型(linestyle)为虚线.
        plt.plot(x, re, color='red',label='SVM')

        plt.xlabel('day')
        plt.ylabel('re')
        # set line syles

        # legend将要显示的信息来自于上面代码中的 label. 所以我们只需要简单写下一下代码, plt 就能自动的为我们添加图例.
        plt.legend(loc='upper right')
        plt.savefig(stock_path+'SVM.png')
        #plt.show()

        #  NB BernoulliNB
        bnb = BernoulliNB()
        bnb.fit(X_train, y_train)  # 训练
        rbnb = bnb.score(X_test, y_test)
        print(" NB BernoulliNB 对X_test,Y_test的 R值(准确率):", rbnb)
        result = bnb.predict(X_test)
        #print("NB BernoulliNB 预测结果：")
        #print(result)
        listBernoulliNBpre = result.tolist()
        re = []
        for i in range(nPredict):
            if listBernoulliNBpre[i] == 1:
                tem = close[i]/ open[i]
                re.append(tem)
            else:
                re.append(1)
        #print(re)
        for i in range(1,nPredict):
            re[i] = re[i-1] * re[i]


        x = np.linspace(0, nPredict, nPredict)
        plt.figure()  # 使用plt.figure定义一个图像窗口.
        plt.plot(x, BuyHold,label='buy-hold')  # 使用plt.plot画(x ,y2)曲线.
        # 使用plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色;
        # 曲线的宽度(linewidth)为1.0；
        # 曲线的类型(linestyle)为虚线.
        plt.plot(x, re, color='red',label='LogisticRegression')

        plt.xlabel('day')
        plt.ylabel('re')
        # set line syles

        # legend将要显示的信息来自于上面代码中的 label. 所以我们只需要简单写下一下代码, plt 就能自动的为我们添加图例.
        plt.legend(loc='upper right')
        plt.savefig(stock_path+'BernoulliNB.png')
        #plt.show()

        #  DecisionTree
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        y_predict = dtc.predict(X_test)
        rdt = dtc.score(X_test, y_test)
        print('DecisionTree 对X_test,Y_test的 R值(准确率):', rdt)
        # print(classification_report(y_predict, y_test, target_names=['died', 'servived']))
        listDecisionTreepre = y_predict.tolist()
        re = []
        for i in range(nPredict):
            if listDecisionTreepre[i] == 1:
                tem = close[i]/ open[i]
                re.append(tem)
            else:
                re.append(1)
        #print(re)
        for i in range(1,nPredict):
            re[i] = re[i-1] * re[i]


        x = np.linspace(0, nPredict, nPredict)
        plt.figure()  # 使用plt.figure定义一个图像窗口.
        plt.plot(x, BuyHold,label='buy-hold')  # 使用plt.plot画(x ,y2)曲线.
        # 使用plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色;
        # 曲线的宽度(linewidth)为1.0；
        # 曲线的类型(linestyle)为虚线.
        plt.plot(x, re, color='red',label='DecisionTree')

        plt.xlabel('day')
        plt.ylabel('re')
        # set line syles

        # legend将要显示的信息来自于上面代码中的 label. 所以我们只需要简单写下一下代码, plt 就能自动的为我们添加图例.
        plt.legend(loc='upper right')
        plt.savefig(stock_path+'DecisionTre.png')
        #plt.show()

        # Random forest
        rf = RandomForestClassifier(max_depth=2, random_state=0)
        rf.fit(X_train, y_train)  # 训练
        rrf = rf.score(X_test, y_test)
        print(" Random forest 对X_test,Y_test的 R值(准确率):", rrf)
        result = rf.predict(X_test)
        # print("Random forest 预测结果：")
        # print(result)
        listRandomForestpre = result.tolist()
        re = []
        for i in range(nPredict):
            if listRandomForestpre[i] == 1:
                tem = close[i]/ open[i]
                re.append(tem)
            else:
                re.append(1)
        #print(re)
        for i in range(1,nPredict):
            re[i] = re[i-1] * re[i]


        x = np.linspace(0, nPredict, nPredict)
        plt.figure()  # 使用plt.figure定义一个图像窗口.
        plt.plot(x, BuyHold,label='buy-hold')  # 使用plt.plot画(x ,y2)曲线.
        # 使用plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色;
        # 曲线的宽度(linewidth)为1.0；
        # 曲线的类型(linestyle)为虚线.
        plt.plot(x, re, color='red',label='RandomForest')

        plt.xlabel('day')
        plt.ylabel('re')
        # set line syles

        # legend将要显示的信息来自于上面代码中的 label. 所以我们只需要简单写下一下代码, plt 就能自动的为我们添加图例.
        plt.legend(loc='upper right')
        plt.savefig(stock_path+'RandomForest.png')
        #plt.show()

        summary = {'KNN':{stock_name:accu}
            ,'LogReg':{stock_name:rls}
            ,'SVM':{stock_name:rsvm}
            ,'Naive Bayes':{stock_name:rbnb}
            ,'Decision Tree':{stock_name:rdt}
            ,'Random Forest':{stock_name:rrf}}
        summ = pd.DataFrame(summary)
        summ.to_excel(stock_path+stock_name+'_summary.xlsx')


        #knn（自己实现）
        result_final = []
        for i in X_test:
            test_class = classify0(i, X_train, y_train, 6)
            result_final.append(test_class)
        # 打印分类结果
        # print(y_test)
        # print(result_final)
        print("knn实现准确率：",accuracy_score(y_test, result_final))

        # 贝叶斯自己实现
        features = [2]
        result_bys = []
        for i in X_test:
            test_class = classify(X_train, y_train, features)
            result_bys.append(test_class)
        print("贝叶斯实现准确率：", accuracy_score(y_test, result_bys))

        #  #逻辑回归自己实现
        # opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradUp'}  # 定义各种可选类型
        # optimalWeights = trainLogRegres(X_train, y_train, opts)                    #权值矩阵
        # accuracy = testLogRegres(optimalWeights, X_test, y_test)
        # print("逻辑回归实现准确率：",accuracy)


        #将各分类器准确率存入表中
        acc_table = pd.DataFrame(columns=('Model','KNN', 'LogReg', 'SVM linear', 'Naive Bayes', 'Decision Tree', 'Random Forest'))




