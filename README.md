# twitterPredict
根据twitter预测股票涨跌

安装

pycharm下直接运行

从yahoo finance爬取历史股价数据，根据提供的每日股票相关twitter信息进行情感分析，通过 vader得到相应的情感得分。将每日的涨跌与情感得分进行匹配。然后将数据80%作为训练集、 20%作为验证集。再通过机器学习，使用sklearn( knn / logistic regression / svm / naïve bayes / decision tree / random forest )，将训练集的情感得分和股票涨跌进行训练。根据训练得到的模型对验证集的股票涨跌进行预测。最后比较根据预测进行买卖操作和根据买入并持有策略哪种方式收益更好，得出根据预测进行买卖收益更好。
