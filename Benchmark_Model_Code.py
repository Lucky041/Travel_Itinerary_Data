import pandas as pd
import numpy as np
import re
import ast
import math
from itertools import chain
import random
import itertools
from datetime import datetime


################Random################
df=pd.read_excel( r"景点间最短时间.xlsx",header=0,index_col=0)
for i in range(len(df)):
    for j in range(len(df)):
        if str(df.loc[df.index[i],df.columns[j]])!='nan':
            dfdf=ast.literal_eval(df.loc[df.index[i],df.columns[j]])
            if len(dfdf)==1 :
                dfdf.append(0.0)
            df.loc[df.index[i],df.columns[j]]=str(dfdf)
df=df.fillna('[0,0]')
POI=pd.read_excel(r"景点.xlsx")
final=pd.DataFrame(columns=['route','value1','value2'])
def Random(routenum,V_usepre):
    totalroutelist=[]
    totalvalue=0
    POI_index=POI.index.tolist()

    for k in range(routenum):
        i= random.choice(POI_index)
        print("起始景点:"+POI.loc[i,'景点名'])
    
        i_time=POI.loc[i,'参观时间']*60*60
        ij_time=0
        j_time=0
        sumtime=i_time+ij_time+j_time
        
        i_value=POI.loc[i,'票价']
        ij_value=0
        j_value=0
        sumvalue=i_value+ij_value+j_value
    
        randomroute_name=[POI.loc[i,'景点名']]
        POI_index.remove(i)
    
        while totalvalue<=V_usepre:
            j=random.choice(POI_index)
        
            ij_time=ast.literal_eval(df.loc[df.index==POI.loc[i,'景点名'],df.columns==POI.loc[j,'景点名']].values[0][0])[0]
            j_time=POI.loc[j,'参观时间']*60*60
            sumtime=sumtime+ij_time+j_time
        
            ij_value=ast.literal_eval(df.loc[df.index==POI.loc[i,'景点名'],df.columns==POI.loc[j,'景点名']].values[0][0])[1]
            j_value=POI.loc[j,'票价']
            sumvalue=sumvalue+ij_value+j_value
            print([sumtime,sumvalue,POI.loc[j,'景点名']])
            #POI_index.remove(j)
        
            if sumtime>10*60*60:
                sumtime=sumtime-ij_time-j_time
                sumvalue=sumvalue-ij_value-j_value
                print("最终时长"+str(sumtime))
                print("最终价格"+str(sumvalue))
                break
            else:
                randomroute_name.append(POI.loc[j,'景点名'])
                POI_index.remove(j)
        
        totalvalue=totalvalue+sumvalue
        print("整体价格"+str(totalvalue))
        totalroutelist.append(randomroute_name)  
    print(totalroutelist)   
    return totalroutelist

top=4#1
userdf=pd.DataFrame()
for l in range(1,101):
    test=pd.read_excel(r"test.xlsx")
    print(str(l)+":开始")
    V_usepre=test.loc[l-1,'人均消费']
    routenum=int(float(test.loc[l-1,'天数'][:-1]))
    llist=[]
    llist1=ast.literal_eval(test.loc[l-1,'路线'])
    for l1 in llist1:
        llist.append(llist1[l1])
    userdf.loc[l-1,'实际路线']=str(llist)
    
    uniqueroute_finallist=[]
    li=0
    while li<top:
        uniqueroute=Random(routenum,V_usepre)
        uniqueroute_finallist.append(uniqueroute)
        li=li+1
    
    userdf.loc[l-1,'预测路线']=str(uniqueroute_finallist)
    print(str(l)+":预测路线")
    print(uniqueroute_finallist)
    
    print(str(l)+":计算命中率")
    ac=[oq for item in llist for oq in item]#真实路线
    a1=0
    for d in ac:
        for dd in range(0,routenum):#旅行天数
            for ddd in range(top):
                if (d in uniqueroute_finallist[ddd][dd]) and (d in llist[dd]):
                    a1=a1+1
                    break
    userdf.loc[l-1,'考虑天数的命中率']=a1/len(set(ac))
    
    print(str(l)+":结束")
    print(userdf)
    #print(userdf.loc[:,"预测路线"].values)
userdf.to_csv("TOP4-Rondom模型.csv")#TOP1-Rondom模型.csv



################POP################
df=pd.read_excel( r"景点间最短时间.xlsx",header=0,index_col=0)
for i in range(len(df)):
    for j in range(len(df)):
        if str(df.loc[df.index[i],df.columns[j]])!='nan':
            dfdf=ast.literal_eval(df.loc[df.index[i],df.columns[j]])
            if len(dfdf)==1 :
                dfdf.append(0.0)
            df.loc[df.index[i],df.columns[j]]=str(dfdf)
df=df.fillna('[0,0]')
POI=pd.read_excel(r"景点.xlsx")
popular_poi=pd.read_excel( r"景点攻略数.xlsx")
popular_poi=popular_poi.sort_values(by='功略数',ascending=False,axis=0)
popular_poi=popular_poi.reset_index(drop=True)
def POP(routenum,V_usepre):
    
    totalroutelist1=[]
    totalvalue1=0
    popular_poi_index=popular_poi.index.tolist()
    for k in range(routenum):
        i= np.min(popular_poi_index)
        print("起始景点:"+str(i)+"_"+popular_poi.loc[i,'景点'])
    
        i_time=POI.loc[POI.景点名==popular_poi.loc[i,'景点'],'参观时间'].values[0]*60*60
        ij_time=0
        j_time=0
        sumtime=i_time+ij_time+j_time
    
        i_value=POI.loc[POI.景点名==popular_poi.loc[i,'景点'],'票价'].values[0]
        ij_value=0
        j_value=0
        sumvalue=i_value+ij_value+j_value
    
        randomroute_name1=[popular_poi.loc[i,'景点']]
        popular_poi_index.remove(i)
    
        while totalvalue1<=V_usepre:
        
            j=np.min(popular_poi_index)
        
            ij_time=ast.literal_eval(df.loc[df.index==popular_poi.loc[i,'景点'],df.columns==popular_poi.loc[j,'景点']].values[0][0])[0]
            j_time=POI.loc[POI.景点名==popular_poi.loc[j,'景点'],'参观时间'].values[0]*60*60
            sumtime=sumtime+ij_time+j_time
        
            ij_value=ast.literal_eval(df.loc[df.index==popular_poi.loc[i,'景点'],df.columns==popular_poi.loc[j,'景点']].values[0][0])[1]
            j_value=POI.loc[POI.景点名==popular_poi.loc[j,'景点'],'票价'].values[0]
            sumvalue=sumvalue+ij_value+j_value
        
            print([sumtime,sumvalue,popular_poi.loc[j,'景点']])
        
            if sumtime>10*60*60:
                sumtime=sumtime-ij_time-j_time
                sumvalue=sumvalue-ij_value-j_value
                print("最终时长"+str(sumtime))
                print("最终价格"+str(sumvalue))
                break
            else:
                randomroute_name1.append(popular_poi.loc[j,'景点'])
                popular_poi_index.remove(j)
            
        totalvalue1=totalvalue1+sumvalue
        print("整体价格"+str(totalvalue1))
    
        totalroutelist1.append(randomroute_name1) 
    return totalroutelist1
    
top=4#1
userdf=pd.DataFrame()
for l in range(1,101):
    test=pd.read_excel(r"test.xlsx")
    print(str(l)+":开始")
    V_usepre=test.loc[l-1,'人均消费']
    routenum=int(float(test.loc[l-1,'天数'][:-1]))
    llist=[]
    llist1=ast.literal_eval(test.loc[l-1,'路线'])
    for l1 in llist1:
        llist.append(llist1[l1])
    userdf.loc[l-1,'实际路线']=str(llist)
    
    uniqueroute_finallist=[]
    uniqueroute=POP(routenum*top,V_usepre)
    for li in range(top):
        uniqueroute_finallist.append(uniqueroute[routenum*li:routenum*li+routenum])
    
    userdf.loc[l-1,'预测路线']=str(uniqueroute_finallist)
    print(str(l)+":预测路线")
    print(uniqueroute_finallist)
    
    print(str(l)+":计算命中率")
    ac=[oq for item in llist for oq in item]#真实路线
    a1=0
    for d in ac:
        for dd in range(0,routenum):#旅行天数
            for ddd in range(top):
                if (d in uniqueroute_finallist[ddd][dd]) and (d in llist[dd]):
                    a1=a1+1
                    break
    userdf.loc[l-1,'考虑天数的命中率']=a1/len(set(ac))
    
    print(str(l)+":结束")
    print(userdf)
    #print(userdf.loc[:,"预测路线"].values)
userdf.to_csv("TOP4-POP模型.csv")#TOP1-POP模型.csv


################NSM################
df=pd.read_excel( r"景点间最短时间.xlsx",header=0,index_col=0)
for i in range(len(df)):
    for j in range(len(df)):
        if str(df.loc[df.index[i],df.columns[j]])!='nan':
            dfdf=ast.literal_eval(df.loc[df.index[i],df.columns[j]])
            if len(dfdf)==1 :
                dfdf.append(0.0)
            df.loc[df.index[i],df.columns[j]]=str(dfdf)
df=df.fillna('[0,0]')
POI=pd.read_excel(r"景点.xlsx")
def NSM(routenum,V_usepre):
    totalroutelist3=[]
    totalvalue3=0
    POI_index=POI.index.tolist()

    for k in range(routenum):
        i= random.choice(POI_index)
        print("起始景点:"+POI.loc[i,'景点名'])
    
        i_time=POI.loc[i,'参观时间']*60*60
        ij_time=0
        j_time=0
        sumtime=i_time+ij_time+j_time
    
        i_value=POI.loc[i,'票价']
        ij_value=0
        j_value=0
        sumvalue=i_value+ij_value+j_value
    
        randomroute_name3=[POI.loc[i,'景点名']]
        POI_index.remove(i)
    
        j_poi=POI.loc[i,'景点名']
    
        while totalvalue3<=V_usepre:
        
            ddf=df.loc[df.index==j_poi,:]
            for m in POI.loc[POI_index,'景点名'].tolist():
                ddf.loc[:,m]=ast.literal_eval(ddf.loc[:,m].values[0])[0] 
            ddf=ddf.T
            ddf.drop(ddf[ddf[j_poi].str.contains(pat='[',regex=False)==True].index,inplace=True) #删除包含字符串的行
            ddf=ddf.sort_values(by=j_poi,ascending=True,axis=0)
            j_poi=ddf.index.tolist()[0] #当前 POI 中最近的poi
            #print(ddf.iloc[:5,:])
            #print(j_poi)
        
            ij_time=ast.literal_eval(df.loc[df.index==POI.loc[i,'景点名'],df.columns==j_poi].values[0][0])[0]
            j_time=POI.loc[POI.景点名==j_poi,'参观时间'].values[0]*60*60
            sumtime=sumtime+ij_time+j_time
        
            ij_value=ast.literal_eval(df.loc[df.index==POI.loc[i,'景点名'],df.columns==j_poi].values[0][0])[1]
            j_value=POI.loc[POI.景点名==j_poi,'票价'].values[0]
            sumvalue=sumvalue+ij_value+j_value
            print([sumtime,sumvalue,j_poi])
            #POI_index.remove(j)
        
            if sumtime>10*60*60:
                sumtime=sumtime-ij_time-j_time
                sumvalue=sumvalue-ij_value-j_value
                print("最终时长"+str(sumtime))
                print("最终价格"+str(sumvalue))
                break
            else:
                randomroute_name3.append(j_poi)
                #print(POI_index)
                #print(POI.loc[POI.景点名==j_poi,:].index[0])
                POI_index.remove(POI.loc[POI.景点名==j_poi,:].index[0])
        
        totalvalue3=totalvalue3+sumvalue
        print("整体价格"+str(totalvalue3))
        totalroutelist3.append(randomroute_name3)  
    return totalroutelist3

top=4#1
userdf=pd.DataFrame()
for l in range(1,101):
    test=pd.read_excel(r"test.xlsx")
    print(str(l)+":开始")
    V_usepre=test.loc[l-1,'人均消费']
    routenum=int(float(test.loc[l-1,'天数'][:-1]))
    llist=[]
    llist1=ast.literal_eval(test.loc[l-1,'路线'])
    for l1 in llist1:
        llist.append(llist1[l1])
    userdf.loc[l-1,'实际路线']=str(llist)
    
    uniqueroute_finallist=[]
    i=0
    while i<top:
        uniqueroute=NSM(routenum,V_usepre)
        uniqueroute_finallist.append(uniqueroute)
        i=i+1
    
    userdf.loc[l-1,'预测路线']=str(uniqueroute_finallist)
    print(str(l)+":预测路线")
    print(uniqueroute_finallist)
    
    print(str(l)+":计算命中率")
    ac=[oq for item in llist for oq in item]#真实路线
    a1=0
    for d in ac:
        for dd in range(0,routenum):#旅行天数
            for ddd in range(top):
                if (d in uniqueroute_finallist[ddd][dd]) and (d in llist[dd]):
                    a1=a1+1
                    break
    userdf.loc[l-1,'考虑天数的命中率']=a1/len(set(ac))
    
    print(str(l)+":结束")
    print(userdf)
    #print(userdf.loc[:,"预测路线"].values)
userdf.to_csv("TOP4-NSM模型.csv")#TOP1-NSM模型.csv


################Item-CF################
lines = open('2_train.txt').readlines()
fp = open('2_train1.txt','w')
for s in lines:
    fp.write(s.replace(',','\t'))
fp.close()

import random
import operator
import pandas as pd

class ItemBasedCF:
    def __init__(self):
        self.N = {} 
        self.W = {} 
        self.train1 = {} 
        self.k = 100 
        self.n = 50  

    
    def get_data(self, file_path):
        """
        @description: load data from file
        @param file_path: path of file
        """
        print('start loading data from ', file_path)
        with open(file_path, "r") as f:
            for i, line in enumerate(f, 0): # remove the first line that is title
                line = line.strip('\r')
                user, item, rating, timestamp = line.split('\t')
                self.train1.setdefault(user, [])
                self.train1[user].append([item, rating])
        print('loading data successfully')
                

    def similarity(self):
        """
        @description: caculate similarity between item i and item j
        """
        print('start caculating similarity matrix ...')
        for user, item_ratings in self.train1.items():
            items = [x[0] for x in item_ratings]    # items that user have interacted 
            for i in items:
                self.N.setdefault(i, 0)
                self.N[i] += 1  # number of users who have interacted item i  
                for j in items:
                    if i != j:
                        self.W.setdefault(i, {})
                        self.W[i].setdefault(j, 0)
                        self.W[i][j] += 1   # number of users who have interacted item i and item j
                    else:
                        self.W.setdefault(i, {})
                        self.W[i].setdefault(j, 0)
                        self.W[i][j] += 0
                        
        for i, j_cnt in self.W.items():
            for j, cnt in j_cnt.items():
                self.W[i][j] = self.W[i][j] / (self.N[i] * self.N[j]) ** 0.5    # similarity between item i and item j
        print('caculating similarity matrix successfully')
      

    def recommendation(self, user):
        """
        @description: recommend n item for user
        @param user: recommended user
        @return items recommended for user
        """
        print('start recommending items for user whose userId is ', user)
        rank = {}
        watched_items=[]
        for xx in self.train1[user]:
            watched_items.append(xx[0])
        for i in watched_items:
            for j, similarity in sorted(self.W[i].items(), key=operator.itemgetter(1), reverse=True)[0:self.k]:
                if j not in watched_items:
                    rank.setdefault(j, 0.)
                    rank[j] += float(self.train1[user][watched_items.index(i)][1]) * similarity  # rating that user rate for item i * similarity between item i and item j
        return sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:self.n]

file_path = "2_train1.txt"
itemBasedCF = ItemBasedCF()
itemBasedCF.get_data(file_path)
itemBasedCF.similarity()
iidf=pd.DataFrame(index=[i for i in range(901,1001)])
for i in range(901,1001):
    user=[str(i)]
    rec = itemBasedCF.recommendation(user[0])
    print('\nitems recommeded for user whose userId is', user[0], ':')
    print(rec)
    reclist=[]
    for j in rec:
        reclist.append(int(j[0]))
    iidf.loc[i,"Topk推荐:"]=str(reclist)

iidf.to_excel(excel_writer = r"ItemCF.xlsx")

iidf=pd.read_excel(r"ItemCF.xlsx",index_col=0)
popular_poi=pd.read_excel( r"景点攻略数.xlsx",index_col=0)
popular_poi=popular_poi.sort_values(by='功略数',ascending=False,axis=0)
popular_poi=popular_poi.reset_index(drop=True)

df=pd.read_excel( r"景点间最短时间.xlsx",header=0,index_col=0)
POI=pd.read_excel(r"景点.xlsx")
for i in range(len(df)):
    for j in range(len(df)):
        if str(df.loc[df.index[i],df.columns[j]])!='nan':
            dfdf=ast.literal_eval(df.loc[df.index[i],df.columns[j]])
            if len(dfdf)==1 :
                dfdf.append(0.0)
            df.loc[df.index[i],df.columns[j]]=str(dfdf)
df=df.fillna('[0,0]')

def ItemCF(a,routenum,V_usepre):
    totalroutelist5=[]
    totalvalue5=0
    popular_poi_index=popular_poi.index.tolist()
    
    for k in range(routenum):
        if len(a)!=0:
            i= a[0]-1
            print("起始景点:"+str(i)+"_"+POI.loc[i,'景点名'])
            i_time=POI.loc[i,'参观时间']*60*60
            ij_time=0
            j_time=0
            sumtime=i_time+ij_time+j_time
    
            i_value=POI.loc[i,'票价']
            ij_value=0
            j_value=0
            sumvalue=i_value+ij_value+j_value
    
            randomroute_name5=[POI.loc[i,'景点名']]
            a.remove(i+1)
        
        else:
            i= np.min(popular_poi_index)
            print("起始景点:"+str(i)+"_"+popular_poi.loc[i,'景点'])
            i_time=POI.loc[POI.景点名==popular_poi.loc[i,'景点'],'参观时间'].values[0]*60*60
            ij_time=0
            j_time=0
            sumtime=i_time+ij_time+j_time
    
            i_value=POI.loc[POI.景点名==popular_poi.loc[i,'景点'],'票价'].values[0]
            ij_value=0
            j_value=0
            sumvalue=i_value+ij_value+j_value
    
            randomroute_name5=[popular_poi.loc[i,'景点']]
            popular_poi_index.remove(i)
            
    
        while totalvalue5<=V_usepre:
            
            if len(a)!=0:
                j=a[0]-1
                ij_time=ast.literal_eval(df.loc[df.index==POI.loc[i,'景点名'],df.columns==POI.loc[j,'景点名']].values[0][0])[0]
                j_time=POI.loc[j,'参观时间']*60*60
                sumtime=sumtime+ij_time+j_time
        
                ij_value=ast.literal_eval(df.loc[df.index==POI.loc[i,'景点名'],df.columns==POI.loc[j,'景点名']].values[0][0])[1]
                j_value=POI.loc[j,'票价']
                sumvalue=sumvalue+ij_value+j_value
                print([sumtime,sumvalue,POI.loc[j,'景点名']])
                xxxx=POI.loc[j,'景点名']
                
                
            else:
                j=np.min(popular_poi_index)
                ij_time=ast.literal_eval(df.loc[df.index==popular_poi.loc[i,'景点'],df.columns==popular_poi.loc[j,'景点']].values[0][0])[0]
                j_time=POI.loc[POI.景点名==popular_poi.loc[j,'景点'],'参观时间'].values[0]*60*60
                sumtime=sumtime+ij_time+j_time
        
                ij_value=ast.literal_eval(df.loc[df.index==popular_poi.loc[i,'景点'],df.columns==popular_poi.loc[j,'景点']].values[0][0])[1]
                j_value=POI.loc[POI.景点名==popular_poi.loc[j,'景点'],'票价'].values[0]
                sumvalue=sumvalue+ij_value+j_value
                print([sumtime,sumvalue,popular_poi.loc[j,'景点']])
                xxxx=popular_poi.loc[j,'景点']
                
            if sumtime>10*60*60:
                sumtime=sumtime-ij_time-j_time
                sumvalue=sumvalue-ij_value-j_value
                print("最终时常"+str(sumtime))
                print("最终价格"+str(sumvalue))
                break
            else:
                randomroute_name5.append(xxxx)
                if len(a)!=0:
                    a.remove(j+1)
                else:   
                    popular_poi_index.remove(j)
            
        totalvalue5=totalvalue5+sumvalue
        print("整体价格"+str(totalvalue5))
    
        totalroutelist5.append(randomroute_name5)
    return totalroutelist5
    
top=4#1
userdf=pd.DataFrame()
for l in range(1,101):
    test=pd.read_excel(r"test.xlsx")
    print(str(l)+":开始")
    V_usepre=test.loc[l-1,'人均消费']
    routenum=int(float(test.loc[l-1,'天数'][:-1]))
    llist=[]
    llist1=ast.literal_eval(test.loc[l-1,'路线'])
    for l1 in llist1:
        llist.append(llist1[l1])
    userdf.loc[l-1,'实际路线']=str(llist)
    a=ast.literal_eval(iidf.loc[l+900,'Topk推荐:'])
    
    uniqueroute_finallist=[]
    uniqueroute=ItemCF(a,routenum*top,V_usepre)
    for li in range(top):
        uniqueroute_finallist.append(uniqueroute[routenum*li:routenum*li+routenum])
    
    userdf.loc[l-1,'预测路线']=str(uniqueroute_finallist)
    print(str(l)+":预测路线")
    print(uniqueroute_finallist)
    
    print(str(l)+":计算命中率")
    ac=[oq for item in llist for oq in item]#真实路线
    a1=0
    for d in ac:
        for dd in range(0,routenum):#旅行天数
            for ddd in range(top):
                if (d in uniqueroute_finallist[ddd][dd]) and (d in llist[dd]):
                    a1=a1+1
                    break
    userdf.loc[l-1,'考虑天数的命中率']=a1/len(set(ac))
    
    print(str(l)+":结束")
    print(userdf)
    print(userdf.loc[:,"预测路线"].values)
userdf.to_csv("TOP4-Item-CF模型.csv")#TOP1-Item-CF模型.csv


################BPR-MF################
Tourism=pd.read_excel(r"train.xlsx")
user_dictlist=[]
for i in range(len(Tourism)):
    user_dict=ast.literal_eval(Tourism.loc[i,'路线'])
    for j in list(user_dict.keys()):
        user_dictlist.append(j)
routelist=np.unique(user_dictlist)
for k in range(len(routelist)):
    Tourism.loc[:,routelist[k]]=""    
for i in range(len(Tourism)):
    for k in range(len(routelist)):
        Tourism.loc[i,routelist[k]]=str(ast.literal_eval(Tourism.loc[i,'路线']).get(routelist[k]))[1:-1]
        Tourism.loc[i,routelist[k]]=Tourism.loc[i,routelist[k]].replace(' ','')

data1=pd.read_excel(r"数据.xlsx")
for i in range(len(Tourism)):
    for j in range(len(data1)):
        if Tourism.loc[i,'标题']==data1.loc[j,'标题']:
            match = re.search(r'\d{4}-\d{2}-\d{2}', data1.loc[j,'用户'])
            Tourism.loc[i,'时间点']=datetime.strptime(match.group(), '%Y-%m-%d')
            
Tourism1=pd.read_excel(r"test.xlsx")
user_dictlist=[]
for i in range(len(Tourism1)):
    user_dict=ast.literal_eval(Tourism1.loc[i,'路线'])
    for j in list(user_dict.keys()):
        user_dictlist.append(j)
routelist=np.unique(user_dictlist)
for k in range(len(routelist)):
    Tourism1.loc[:,routelist[k]]=""    
for i in range(len(Tourism1)):
    for k in range(len(routelist)):
        Tourism1.loc[i,routelist[k]]=str(ast.literal_eval(Tourism1.loc[i,'路线']).get(routelist[k]))[1:-1]
        Tourism1.loc[i,routelist[k]]=Tourism1.loc[i,routelist[k]].replace(' ','')

data2=pd.read_excel(r"数据.xlsx")
for i in range(len(Tourism1)):
    for j in range(len(data2)):
        if Tourism1.loc[i,'标题']==data2.loc[j,'标题']:
            match = re.search(r'\d{4}-\d{2}-\d{2}', data2.loc[j,'用户'])
            Tourism1.loc[i,'时间点']=datetime.strptime(match.group(), '%Y-%m-%d')

Tourism.to_excel(excel_writer = r"train1.xlsx")
Tourism1.to_excel(excel_writer = r"test1.xlsx")

Tourism=pd.read_excel("train1.xlsx",index_col=0)
Tourism1=pd.read_excel("test1.xlsx",index_col=0)

#'user'+'\t'+'item'+'\t'+'rating'+'\t'+'time'+'\n'
file_handle=open('2_train.txt',mode='w')
file_handle1=open('2_test.txt',mode='w')
#file_handle.write('user'+'\t'+'item'+'\t'+'rating'+'\t'+'time'+'\n')
for i in range(0,len(Tourism)):
    print(i)
    strlist=[]
    a=Tourism.iloc[i,9:-1].tolist()
    a=list(filter(lambda x : x != 'on', a))
    alist=[]
    for j in a:
        b=ast.literal_eval(j)
        if type(b)==str:
            alist.append([b])
        else:
            alist.append(list(b))
    alist= list(itertools.chain.from_iterable(alist))
    print(a)
    for k in alist:
        file_handle.write(str(i+1)+","+str(POI.loc[POI.景点名==k,:].index[0]+1)+","+str(1)+","+str(Tourism.iloc[i,-1])+'\n')

for ii in range(0,len(Tourism1)):
    print(ii)
    strlist1=[]
    aa=Tourism1.iloc[ii,10:-1].tolist()
    aa=list(filter(lambda xx : xx != 'on', aa))
    alist1=[]
    for jj in aa:
        bb=ast.literal_eval(jj)
        if type(bb)==str:
            alist1.append([bb])
        else:
            alist1.append(list(bb))
    alist1= list(itertools.chain.from_iterable(alist1))
    print(aa)
    lll = 0
    while True:
        lll=random.randint(0,len(alist1)-1)
        if str(POI.loc[POI.景点名==alist1[lll],:].index.values)!='[]':
            break
    file_handle.write(str(ii+901)+","+str(POI.loc[POI.景点名==alist1[lll],:].index[0]+1)+","+str(1)+","+str(Tourism1.iloc[ii,-1])+'\n')
    
    lll1=0
    while True:
        lll1=random.randint(0,len(alist1)-1)
        if len(alist1)==1:
            break
        else:
            if lll1!=lll and str(POI.loc[POI.景点名==alist1[lll1],:].index.values)!='[]':
                break
    file_handle1.write(str(ii+901)+","+str(POI.loc[POI.景点名==alist1[lll1],:].index[0]+1)+","+str(1)+","+str(Tourism1.iloc[ii,-1])+'\n')

file_handle.close() 
file_handle1.close()

import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 

def getUI(dsname,dformat): 
    train = pd.read_csv(dsname+'_train.txt',header = None,names = dformat,sep = ',')
    test = pd.read_csv(dsname+'_test.txt',header = None,names = dformat,sep = ',')
    rating = pd.concat([train,test])
    print(rating)
    all_user = np.unique(rating['user'])
    num_user = max(all_user)#len(all_user)
    all_item = np.unique(rating['item'])
    num_item = max(all_item)#len(np.unique(rating['item']))
    train.sort_values(by=['user','time'],axis=0,inplace=True) 
    #train.to_csv('./train.txt',index = False,header=0)
    test.sort_values(by=['user','time'],axis=0,inplace=True)
    #test.to_csv('./test.txt',index = False,header=0)
    return all_user,all_item,train,test   

def train(train_data,all_user,all_item,step,iters,dimension):
#    userVec,itemVec=initVec(all_user,all_item,dimension)
    user_cnt = max(all_user)+1
    item_cnt = max(all_item)+1
    userVec = np.random.uniform(0,1,size=(user_cnt,dimension))#*0.01
    itemVec = np.random.uniform(0,1,size=(item_cnt,dimension))#*0.01
    lr = 0.005
    reg = 0.01
    f = open('train_record.txt','w')
    sum_t = 0
    train_user = np.unique(train_data['user'])
    L = []
    for i in range(iters):
        loss = 0
        cnt = 0
        st = time.time()
        for j in range(step):
            user = int(np.random.choice(train_user,1))
            visited =  np.array(train_data[train_data['user']==user]['item'])
            # print(visited)
            itemI = int(np.random.choice(visited,1))
            itemJ = int(np.random.choice(all_item,1))
            while itemJ in visited:
                itemJ = int(np.random.choice(all_item,1))
            # print(user,itemI,itemJ)
            # print(userVec[user],userVec[user].shape)
            # print(itemVec[itemI],itemVec[itemI].shape)
            r_ui = np.dot(userVec[user], itemVec[itemI].T) 
            r_uj = np.dot(userVec[user], itemVec[itemJ].T) 
            r_uij = r_ui - r_uj
            factor = 1.0 / (1 + np.exp(r_uij))   
            # update U and V
            userVec[user] += lr * (factor * (itemVec[itemI] - itemVec[itemJ]) + reg * userVec[user])
            itemVec[itemI] += lr * (factor * userVec[user] + reg * itemVec[itemI])
            itemVec[itemJ] += lr * (factor * (-userVec[user]) + reg * itemVec[itemJ])
            
            loss += (1.0 / (1 + np.exp(-r_uij)))

            cnt = cnt + 1
        # loss = 1.0 * loss / cnt
        loss += + reg * (
                    np.power(np.linalg.norm(userVec,ord=2),2) 
                    + np.power(np.linalg.norm(itemVec,ord=2),2) 
                    + np.power(np.linalg.norm(itemVec,ord=2),2)
                    )
        L.append(loss)
        pt = time.time()-st
        sum_t = sum_t + pt
        print("The "+str(i+1)+" is done! cost time:",pt)
        print("Loss:",loss)
        f.write("The "+str(i+1)+" is done! cost time:"+str(pt))
    L = np.array(L)
    plt.plot(np.arange(iters),L,c = 'r',marker = 'o') 
    print("\nLoss图像为：")
    plt.show()
    f.write('Training cost time: '+str(sum_t))    
    f.close()
    np.savetxt('userVec.txt',userVec,delimiter=',',newline='\n')
    np.savetxt('itemVec.txt',itemVec,delimiter=',',newline='\n')

def topk(dic,k):
    keys = []
    values = []
    for i in range(0,k):
        key,value = max(dic.items(),key=lambda x: x[1])
        keys.append(key)
        values.append(value)
        dic.pop(key)
    return keys,values

def cal_indicators(rankedlist, testlist):
    HITS_i = 0
    sum_precs = 0
    AP_i = 0 
    len_R = 0 
    len_T = 0
    MRR_i = 0 

    ranked_score = []
    for n in range(len(rankedlist)):
        if rankedlist[n] in testlist:
            HITS_i += 1
            sum_precs += HITS_i / (n + 1.0)
            if MRR_i == 0:
                MRR_i = 1.0/(rankedlist.index(rankedlist[n])+1)
                
        else:
            ranked_score.append(0)
    if HITS_i > 0:
        AP_i = sum_precs/len(testlist)
    len_R = len(rankedlist)
    len_T = len(testlist)
    return AP_i,len_R,len_T,MRR_i,HITS_i

def test(all_user,all_item,train_data,test_data,dimension,k):
    userP = np.loadtxt('userVec.txt',delimiter=',',dtype=float)
    itemP = np.loadtxt('itemVec.txt',delimiter=',',dtype=float)
    PRE = 0
    REC = 0
    MAP = 0
    MRR = 0
    
    AP = 0
    HITS = 0
    sum_R = 0
    sum_T = 0
    valid_cnt = 0
    stime = time.time()
    test_user = np.unique(test_data['user'])
    idf=pd.DataFrame(index=test_user)
    for user in test_user:
#        user = 0
        visited_item = list(train_data[train_data['user']==user]['item'])
        #print('访问过的item:',visited_item)
        if len(visited_item)==0: 
            continue
        print("对用户",user)
        per_st = time.time()
        testlist = list(test_data[test_data['user']==user]['item'].drop_duplicates()) 
        #testlist = list(set(testlist)-set(testlist).intersection(set(visited_item))) 
        if len(testlist)==0: 
            continue
        print("对用户",user)
        print([testlist])
        valid_cnt = valid_cnt + 1 
        poss = {}        
        for item in all_item:
            if item in visited_item:
                continue
            else:
                poss[item] = np.dot(userP[user],itemP[item])
#        print(poss)
        rankedlist,test_score = topk(poss,k) 
        print("Topk推荐:",rankedlist)
        print("实际访问:",testlist)
        idf.loc[user,'Topk推荐:']=str(rankedlist)
#        print("单条推荐耗时:",time.time() - per_st)
        AP_i,len_R,len_T,MRR_i,HITS_i= cal_indicators(rankedlist, testlist)
        AP += AP_i
        sum_R += len_R
        sum_T += len_T
        MRR += MRR_i
        HITS += HITS_i
#        print(test_score)
#        print('--------')
#        break
    etime = time.time()
    PRE = HITS/(sum_R*1.0)
    REC = HITS/(sum_T*1.0)
    MAP = AP/(valid_cnt*1.0)
    MRR = MRR/(valid_cnt*1.0)
    p_time = (etime-stime)/valid_cnt
    
    print('评价指标如下:')
    print('PRE@',k,':',PRE)
    print('REC@',k,':',REC)
    print('MAP@',k,':',MAP)
    print('MRR@',k,':',MRR)
    print('平均每条推荐耗时:',p_time)
    print('总耗时：',etime-stime)
    with open('result_'+dsname[-1]+'.txt','w') as f:
        f.write('评价指标如下:\n')
        f.write('PRE@'+str(k)+':'+str(PRE)+'\n')
        f.write('REC@'+str(k)+':'+str(REC)+'\n')
        f.write('MAP@'+str(k)+':'+str(MAP)+'\n')
        f.write('MRR@'+str(k)+':'+str(MRR)+'\n') 
        f.write('平均每条推荐耗时@:'+str(k)+':'+str(p_time)+'\n') 
        f.write('总耗时@:'+str(k)+':'+str(etime-stime)+'s\n') 
    return idf  

dsname = '2'
dformat = ['user','item','rating','time']
iters = 30
step = 10000
all_user,all_item,train_data,test_data = getUI(dsname,dformat)
dimension = 60
train(train_data,all_user,all_item,step,iters,dimension)
k = 100   

k = 100
idf=test(all_user,all_item,train_data,test_data,dimension,k)
idf.to_excel(excel_writer = r"BPR-MF.xlsx")
userP = np.loadtxt('userVec.txt',delimiter=',',dtype=float)
itemP = np.loadtxt('itemVec.txt',delimiter=',',dtype=float)

idf=pd.read_excel(r"BPR-MF.xlsx",index_col=0)
POI=pd.read_excel(r"景点.xlsx")
df=pd.read_excel( r"景点间最短时间.xlsx",header=0,index_col=0)
for i in range(len(df)):
    for j in range(len(df)):
        if str(df.loc[df.index[i],df.columns[j]])!='nan':
            dfdf=ast.literal_eval(df.loc[df.index[i],df.columns[j]])
            if len(dfdf)==1 :
                dfdf.append(0.0)
            df.loc[df.index[i],df.columns[j]]=str(dfdf)
df=df.fillna('[0,0]')
def BPR(a,routenum,V_usepre):
    totalroutelist4=[]
    totalvalue4=0

    for k in range(routenum):
        i= a[0]-1
        print("起始景点:"+str(i)+"_"+POI.loc[i,'景点名'])
    
        i_time=POI.loc[i,'参观时间']*60*60
        ij_time=0
        j_time=0
        sumtime=i_time+ij_time+j_time
    
        i_value=POI.loc[i,'票价']
        ij_value=0
        j_value=0
        sumvalue=i_value+ij_value+j_value
    
        randomroute_name4=[POI.loc[i,'景点名']]
        a.remove(i+1)
    
        while totalvalue4<=V_usepre:
        
            j=a[0]-1
        
            ij_time=ast.literal_eval(df.loc[df.index==POI.loc[i,'景点名'],df.columns==POI.loc[j,'景点名']].values[0][0])[0]
            j_time=POI.loc[j,'参观时间']*60*60
            sumtime=sumtime+ij_time+j_time
        
            ij_value=ast.literal_eval(df.loc[df.index==POI.loc[i,'景点名'],df.columns==POI.loc[j,'景点名']].values[0][0])[1]
            j_value=POI.loc[j,'票价']
            sumvalue=sumvalue+ij_value+j_value
        
            print([sumtime,sumvalue,POI.loc[j,'景点名']])
        
            if sumtime>10*60*60:
                sumtime=sumtime-ij_time-j_time
                sumvalue=sumvalue-ij_value-j_value
                print("最终时常"+str(sumtime))
                print("最终价格"+str(sumvalue))
                break
            else:
                randomroute_name4.append(POI.loc[j,'景点名'])
                a.remove(j+1)
            
        totalvalue4=totalvalue4+sumvalue
        print("整体价格"+str(totalvalue4))
    
        totalroutelist4.append(randomroute_name4) 
    return totalroutelist4
    
top=4#1
userdf=pd.DataFrame()
for l in range(1,101):
    test=pd.read_excel(r"test.xlsx")
    print(str(l)+":开始")
    V_usepre=test.loc[l-1,'人均消费']
    routenum=int(float(test.loc[l-1,'天数'][:-1]))
    llist=[]
    llist1=ast.literal_eval(test.loc[l-1,'路线'])
    for l1 in llist1:
        llist.append(llist1[l1])
    userdf.loc[l-1,'实际路线']=str(llist)
    a=ast.literal_eval(idf.loc[l+900,'Topk推荐:'])
    
    uniqueroute_finallist=[]
    uniqueroute=BPR(a,routenum*top,V_usepre)
    for li in range(top):
        uniqueroute_finallist.append(uniqueroute[routenum*li:routenum*li+routenum])
    
    userdf.loc[l-1,'预测路线']=str(uniqueroute_finallist)
    print(str(l)+":预测路线")
    print(uniqueroute_finallist)
    
    print(str(l)+":计算命中率")
    ac=[oq for item in llist for oq in item]#真实路线
    a1=0
    for d in ac:
        for dd in range(0,routenum):#旅行天数
            for ddd in range(top):
                if (d in uniqueroute_finallist[ddd][dd]) and (d in llist[dd]):
                    a1=a1+1
                    break
    userdf.loc[l-1,'考虑天数的命中率']=a1/len(set(ac))
    
    print(str(l)+":结束")
    print(userdf)
    print(userdf.loc[:,"预测路线"].values)
userdf.to_csv("TOP4-BPR-MF模型.csv")#TOP4-BPR-MF模型.csv


################Bin C et al. ################
import pandas as pd
import re
import numpy as np
import ast
import requests
import json
import logging
import time
import copy 
POI=pd.read_excel(r"景点.xlsx")
df=pd.read_excel( r"景点间最短时间.xlsx",header=0,index_col=0)
for i in range(len(df)):
    for j in range(len(df)):
        if str(df.loc[df.index[i],df.columns[j]])!='nan':
            dfdf=ast.literal_eval(df.loc[df.index[i],df.columns[j]])
            if len(dfdf)==1 :
                dfdf.append(0.0)
            df.loc[df.index[i],df.columns[j]]=str(dfdf)
df=df.fillna('[0,0]')

Tourism=pd.read_excel(r"train.xlsx")
user_dictlist=[]
for i in range(len(Tourism)):
    user_dict=ast.literal_eval(Tourism.loc[i,'路线'])
    for j in list(user_dict.keys()):
        user_dictlist.append(j)
routelist=np.unique(user_dictlist)
for k in range(len(routelist)):
    Tourism.loc[:,routelist[k]]=""    
for i in range(len(Tourism)):
    for k in range(len(routelist)):
        Tourism.loc[i,routelist[k]]=str(ast.literal_eval(Tourism.loc[i,'路线']).get(routelist[k]))[1:-1]
        Tourism.loc[i,routelist[k]]=Tourism.loc[i,routelist[k]].replace(' ','')

for o in ['春季','夏季','秋季','冬季']:
    for p in ['亲子','父母','夫妻','朋友','情侣','一个人']:
        print(o,p)
        Candidateroute=Tourism.loc[(Tourism.时间==o) & (Tourism.同行者==p),:]
        Candidateroute=Candidateroute.reset_index(drop=True)
        
        Candidateroutelist=[]
        weightlist=[]
        for ii in range(len(Candidateroute)):
            routes=[]
            for kk in range(len(routelist)):
                if Candidateroute.loc[ii,routelist[kk]]!='on':
                    result_list = re.split(",",Candidateroute.loc[ii,routelist[kk]])
                    routes.append(result_list)
            routess=[m for item in routes for m in item]
            if routess!=[]:
                Candidateroutelist.append(routess) 
                weightlist.append(1)
                
        totalCandidateroutelist=[]
        for i in range(len(Candidateroutelist)):
            Medianvalue=[]
            #print(Candidateroutelist[i])
            for j in range(len(Candidateroutelist[i])-1):
                #print(Candidateroutelist[i][j][1:-1],POI.loc[POI.景点名==Candidateroutelist[i][j][1:-1],'参观时间'])
                #print(Candidateroutelist[i][j][1:-1],Candidateroutelist[i][j+1][1:-1])
                Medianvalue.append([Candidateroutelist[i][j][1:-1],POI.loc[POI.景点名==Candidateroutelist[i][j][1:-1],'参观时间'].values[0]])
                Medianvalue.append([list(map(float,re.findall("\d+\.?\d*",df.loc[Candidateroutelist[i][j][1:-1],Candidateroutelist[i][j+1][1:-1]])))[0]])
            #print(Candidateroutelist[i][len(Candidateroutelist[i])-1][1:-1],POI.loc[POI.景点名==Candidateroutelist[i][len(Candidateroutelist[i])-1][1:-1],'参观时间'])
            Medianvalue.append([Candidateroutelist[i][len(Candidateroutelist[i])-1][1:-1],POI.loc[POI.景点名==Candidateroutelist[i][len(Candidateroutelist[i])-1][1:-1],'参观时间'].values[0]])
            totalCandidateroutelist.append(Medianvalue)
            
        dftotal=pd.DataFrame(index=[i for i in range(len(totalCandidateroutelist))], columns=['路线','权重'])
        for iii in range(len(totalCandidateroutelist)):
            dftotal.loc[iii,'路线']=totalCandidateroutelist[iii]
        dftotal.loc[:,'权重']=weightlist
        dftotal.to_excel(excel_writer = r"候选路线及权重_"+o+p+".xlsx")

def getElem(dataList):      
    elem = []
    for i in dataList[:]:
        for j in i[:]:
            if len(j)>1:
                if j not in elem:      
                    elem.append(j)

    elem = sorted(elem)    
    return elem

def useCycleGetFreElem(dataList, prefixE, elem, minsup,weight):     
    elemsup = {}    
    for e in elem: 
        for i in dataList[:]:
            x = dataList.index(i)
            for j in i:
                if e[0] in j:
                    #print(weight[x])
                    elemsup[str(e)] = elemsup.get(str(e), 0) + 1*weight[x] 1
                    break
    #print(elemsup)
    freElem = []
    notFreElem = []
    #print(elemsup)
    for i in elemsup.keys():
        if elemsup[i] >= minsup:  
            freElem.append(eval(i))
        else:
            notFreElem.append(eval(i))
    #print(freElem)
    #print(notFreElem)
    return freElem, notFreElem

def deleteNotFreElem(data, notFreElem):     
    if len(notFreElem) == 0:
        return
    for i in data[:]:
        for j in i[:]:
            if len(j)>=1 and j in notFreElem:
                x = data.index(i)
                y = i.index(j)
                data[x][y]=[] 
                if y!=len(i[:])-1:
                    data[x][y+1]=[]  
                if (y!=len(i[:])-1)& (y!=0):
                    data[x][y-1]=[list(map(float,re.findall("\d+\.?\d*",df.loc[data[x][y-2][0],data[x][y+2][0]])))[0]]
                while [] in i:
                    i.remove([])     
        if i!=[] and len(i[-1])==1:
            i[-1]=[]
        while [] in i:
            i.remove([])      
    while [] in data:
        data.remove([])      
    return

def getPrefixData(e, data): 
    copyData = list(copy.deepcopy(data))    
    #print(e)                                           
    flage = 0                 
    for i in copyData[:]:
        x = copyData.index(i)
        for j in i[:]:
            y = i.index(j)
            if e != j:
                copyData[x][y]=[]  
            else:
                copyData[x][y]=[]
                if y!=len(i[:])-1:
                    copyData[x][y+1]=[]
                flage = 1   
                break
        while [] in i:
            i.remove([])            
    while [] in copyData:
        copyData.remove([])           
    return copyData 

def getAllPrefixData( elem, prefixE, dataList):
    data1 = list(copy.deepcopy(dataList))   
    #print(data1)
    allPrefixData = []  
    for e in elem:
        temp2 = getPrefixData( e,data1)
        allPrefixData.append( temp2 )   
        #print(allPrefixData)
    return allPrefixData  

def cycleGetFreElem(preFixData, e, minsup,weight):    
    copyPreFixData = list(copy.deepcopy(preFixData))
    allFreSequence = [  ]   
    allElem = getElem(copyPreFixData)  
    #print(allElem)
    
    freElem, notFreElem = useCycleGetFreElem(copyPreFixData, e, allElem, minsup,weight)    
    #print(notFreElem)
    deleteNotFreElem(copyPreFixData, notFreElem)   
    #print("++++++++++++++++++++++++++++")
    thisAllPrefixData = getAllPrefixData(freElem, e, copyPreFixData)   
    #print(thisAllPrefixData)
    for x in freElem:
        temp2 = [e,x]     
        allFreSequence.append(temp2)      

    lengthFreElem = len(freElem)
    for i in range(lengthFreElem):
        temp = cycleGetFreElem(thisAllPrefixData[i], freElem[i], minsup,weight)  
        for x in temp:  
            t2 = copy.deepcopy(x)
            t2.insert(0, e)   
            allFreSequence.append(t2)

        #allFreElem.append(list(temp))
    #print(allFreSequence)
    return allFreSequence   

def prefixSpan(dataList, minsup,weight): 
    dd=copy.deepcopy(dataList)
    #print(dataList)
    elem = getElem(dataList)  
    freElem, notFreElem = useCycleGetFreElem(dataList,'-1', elem, minsup,weight)   
    #print(notFreElem)         #  ['a', 'b', 'c', 'd', 'e', 'f'] ['g']
    deleteNotFreElem(dataList, notFreElem)     
    #print(dataList1)
    
    allPrefixData = getAllPrefixData(freElem, '-1' , dataList)    
    
    allfreSequence = {}     
    allListFreSequence = []     
    lengthFreElem = len(allPrefixData)
    #print(lengthFreElem)
    
    for x in range(lengthFreElem):
        l = cycleGetFreElem(allPrefixData[x], freElem[x], minsup,weight)   
        l.insert(0, [freElem[x]])    
        allfreSequence[str(freElem[x])] = l
        allListFreSequence.append(l)
   
    toalroutelist=[]
    fretoalroute=[]
    for lengthE in range(lengthFreElem):    
        print(freElem[lengthE],'--------------->>>>>>>>>>')
        #print(allListFreSequence[lengthE])
        for x in allListFreSequence[lengthE]:
            qmsup = {}   
            for i in dd[:]:
                x_weight= dd.index(i)
                ii=set(tuple(tt) for tt in i)
                xx=set(tuple(tt) for tt in x)
                if xx.issubset(ii):
                    #print(str(i))
                    iindex=0
                    inum=0
                    iii=i
                    for q in x:
                        iii=iii[iindex:]
                        #print(i)
                        #print(q)
                        #print(iindex)
                        #print(iii)
                        #print("_______________")
                        if q in iii:
                            inum=inum+1
                            iindex=iii.index(q)+2
                    if inum==len(x):
                        qmsup[str(x)] = qmsup.get(str(x), 0) + 1*weight[x_weight] 
            #print("路线频率为：-------------------------------------")
            #print(qmsup)
            fretoalroute.append(list(qmsup.values())[0])
            routlist=[]
            for m in x:
                routlist.append(m)
                n=x.index(m)
                if len(x)>1 and n!=len(x)-1:
                    #print(df.loc[x[n][0],x[n+1][0]])
                    routlist.append(ast.literal_eval(df.loc[x[n][0],x[n+1][0]]))
            print("具体路线为。。。。。。。。。。。。。。。。。。。。")
            print(routlist)
            toalroutelist.append(routlist)
            #print(x)
    toalroutedf=pd.DataFrame()
    toalroutedf['路线']=toalroutelist
    toalroutedf['权重']=fretoalroute
    #print(allfreSequence)
    return allfreSequence,toalroutedf    

for o in ['春季','秋季','冬季']:#'春季','夏季','秋季','冬季'
    for p in ['情侣']:#'亲子','父母','夫妻','朋友','情侣','一个人'
        print(o,p)
        print("______________________________________")
        dftotal=pd.read_excel("候选路线及权重_"+o+p+".xlsx")
        mydata=[ast.literal_eval(dftotal.loc[i,'路线']) for i in range(len(dftotal))]
        weightlist=dftotal.loc[:,'权重'].tolist()
        minsup =len(dftotal)*0.15*np.min(weightlist)
        q,hq = prefixSpan(mydata, minsup,weightlist)
        hq.to_excel(excel_writer = r"PrefixSpan算法结果_"+o+p+".xlsx")

test=pd.read_excel(r"test.xlsx")
for u in range(1,101):#28,33,53,82range(1,101)
    
    print(str(u)+":筛选路线")
    o=test.loc[u-1,'时间']
    p=test.loc[u-1,'同行者']
    PrefixSpanresult=pd.read_excel(r"PrefixSpan算法结果_"+o+p+".xlsx",index_col=0)
    
    for i in range(len(PrefixSpanresult)):
        oneroute=ast.literal_eval(PrefixSpanresult.loc[i,"路线"])
        PrefixSpanresult.loc[i,"总时间"]=0
        if len(oneroute)==1:
            PrefixSpanresult.loc[i,"总时间"]=oneroute[0][1]*60*60
        else:
            totaltime=0
            for j in range(len(oneroute)):
                if (j % 2) == 0:
                    Tvist=oneroute[j][1]*60*60
                    if j==0:
                        Ttraffic_pi1_pi=0
                    else:
                        Ttraffic_pi1_pi=oneroute[j-1][0]
                    
                    if j==len(oneroute)-1:
                        Ttraffic_pi_pi1=0
                    else:
                        Ttraffic_pi_pi1=oneroute[j+1][0] 
                    totaltime=totaltime+Tvist+Ttraffic_pi_pi1
            #print("totaltime:"+str(totaltime))
            PrefixSpanresult.loc[i,"总时间"]=totaltime
    
    n=int(test.loc[u-1,'天数'][:-1])
    if len(PrefixSpanresult)==0:
        PrefixSpanresult=pd.DataFrame(columns=['路线','权重','总时间','总时间','访问时长与总时长比率','季节匹配','总体得分'])
    
    else:
        index = PrefixSpanresult[PrefixSpanresult ["总时间"]> n*12*60*60].index
        PrefixSpanresult.drop(index, axis = 0, inplace=True)
        PrefixSpanresult = PrefixSpanresult.reset_index(drop=True)
        if len(PrefixSpanresult)==0:
            PrefixSpanresult=pd.DataFrame(columns=['路线','权重','总时间','总时间','访问时长与总时长比率','季节匹配','总体得分'])
            
        else:
            print(str(u)+":计算评分")
            for ii in range(len(PrefixSpanresult)):
                oneroute1=ast.literal_eval(PrefixSpanresult.loc[ii,"路线"])
                if len(oneroute1)==1:
                    if str(POI.loc[POI.景点名==oneroute1[0][0],"总评分"].values[0])!="nan":
                        value=POI.loc[POI.景点名==oneroute1[0][0],"总评分"].values[0]
                    else:
                        value=0
                    PrefixSpanresult.loc[ii,"总评分"]=value

                else:
                    value=0
                    for jj in range(len(oneroute1)):
                        if (jj % 2) == 0:
                            if str(POI.loc[POI.景点名==oneroute1[jj][0],"总评分"].values[0])!="nan":
                                value=value+POI.loc[POI.景点名==oneroute1[jj][0],"总评分"].values[0]
                            else:
                                value=value+0 
                    PrefixSpanresult.loc[ii,"总评分"]=value
            PrefixSpanresult.loc[:,"总评分"]=(PrefixSpanresult.loc[:,"总评分"]-np.min(PrefixSpanresult.loc[:,"总评分"]))\
                                            /(np.max(PrefixSpanresult.loc[:,"总评分"])-np.min(PrefixSpanresult.loc[:,"总评分"]))

            print(str(u)+":计算访问时长与总时长比率")
            for iii in range(len(PrefixSpanresult)):
                oneroute2=ast.literal_eval(PrefixSpanresult.loc[iii,"路线"])
                PrefixSpanresult.loc[iii,"访问时长与总时长比率"]=0
                if len(oneroute2)==1:
                    PrefixSpanresult.loc[iii,"访问时长与总时长比率"]=oneroute2[0][1]*60*60/PrefixSpanresult.loc[iii,"总时间"]
                else:
                    totaltime1=0
                    for jjj in range(len(oneroute2)):
                        if (jjj % 2) == 0:
                            Tvist1=oneroute2[jjj][1]*60*60
                            totaltime1=totaltime1+Tvist1
                    PrefixSpanresult.loc[iii,"访问时长与总时长比率"]=totaltime1/PrefixSpanresult.loc[iii,"总时间"]

            print(str(u)+":计算季节匹配")
            season=o
            for iiii in range(len(PrefixSpanresult)):
                oneroute3=ast.literal_eval(PrefixSpanresult.loc[iiii,"路线"])
                if len(oneroute3)==1:
                    if season in POI.loc[POI.景点名==oneroute3[0][0],"适合参观季节"].values[0] or \
                    POI.loc[POI.景点名==oneroute3[0][0],"适合参观季节"].values[0]=="四季皆宜":
                        value1=1/1
                    else:
                        value1=0/1
                    PrefixSpanresult.loc[iiii,"季节匹配"]=value1

                else:
                    value1=0
                    nn=0
                    for jjjj in range(len(oneroute3)):
                        if (jjjj % 2) == 0:
                            nn=nn+1
                            #print(POI.loc[POI.景点名==oneroute3[0][0],"适合参观季节"].values[0])
                            if season in POI.loc[POI.景点名==oneroute3[0][0],"适合参观季节"].values[0] or\
                            POI.loc[POI.景点名==oneroute3[0][0],"适合参观季节"].values[0]=="四季皆宜":
                                value1=value1+1
                            else:
                                value1=value1+0
                    PrefixSpanresult.loc[iiii,"季节匹配"]=value1/nn

            print(PrefixSpanresult)
            print(str(u)+":计算总体得分")
            w1=w2=w3=w4=1/4 
            PrefixSpanresult=PrefixSpanresult.dropna(axis=0,how='any')
            PrefixSpanresult.loc[:,"总体路线得分"]=w1*PrefixSpanresult.iloc[:,3]+w2*PrefixSpanresult.iloc[:,4]+w3*PrefixSpanresult.iloc[:,5]
            PrefixSpanresult=PrefixSpanresult.sort_values(by='总体路线得分',ascending=False,axis=0)
            PrefixSpanresult = PrefixSpanresult.reset_index(drop=True)
        
    PrefixSpanresult.to_excel(excel_writer = r"路线得分//"+str(u)+".xlsx")
    print(str(u)+":完毕")


def X_POI(x):
    totalpoilist=[]
    oneroute=ast.literal_eval(PrefixSpanresult.loc[x,"路线"])
    if (len(oneroute)==1) and (oneroute[0][0] not in totalpoilist):
        totalpoilist.append(oneroute[0][0])
    else:
        for j in range(len(oneroute)):
            if (j % 2) == 0 and (oneroute[j][0] not in totalpoilist):
                totalpoilist.append(oneroute[j][0])
    return totalpoilist


POI=pd.read_excel(r"景点.xlsx")
test=pd.read_excel(r"test.xlsx")
popular_poi=pd.read_excel( r"攻略数.xlsx")
popular_poi=popular_poi.sort_values(by='功略数',ascending=False,axis=0)
popular_poi=popular_poi.reset_index(drop=True)
df=pd.read_excel( r"景点间最短时间.xlsx",header=0,index_col=0)
for i in range(len(df)):
    for j in range(len(df)):
        if str(df.loc[df.index[i],df.columns[j]])!='nan':
            dfdf=ast.literal_eval(df.loc[df.index[i],df.columns[j]])
            if len(dfdf)==1 :
                dfdf.append(0.0)
            df.loc[df.index[i],df.columns[j]]=str(dfdf)
df=df.fillna('[0,0]')


def LUWEN(routenum,totalpoilist):
    totalpoilist_index=[]
    for a in totalpoilist:
        totalpoilist_index.append(totalpoilist.index(a))
    totalroutelist1=[]
    totalvalue1=0
    
    for k in range(routenum):
        if len(totalpoilist_index)==0:
                break
        i= np.min(totalpoilist_index)
        print("起始景点:"+str(i)+"_"+totalpoilist[i])

        i_time=POI.loc[POI.景点名==totalpoilist[i],'参观时间'].values[0]*60*60
        ij_time=0
        j_time=0
        sumtime=i_time+ij_time+j_time

        randomroute_name1=[totalpoilist[i]]
        totalpoilist_index.remove(i)

        while True:
            if len(totalpoilist_index)==0:
                break
            j=np.min(totalpoilist_index)

            ij_time=ast.literal_eval(df.loc[df.index==totalpoilist[i],df.columns==totalpoilist[j]].values[0][0])[0]
            j_time=POI.loc[POI.景点名==totalpoilist[j],'参观时间'].values[0]*60*60
            sumtime=sumtime+ij_time+j_time

            print([sumtime,totalpoilist[j]])

            if sumtime>10*60*60:
                sumtime=sumtime-ij_time-j_time
                print("最终时常"+str(sumtime))
                break
            else:
                randomroute_name1.append(totalpoilist[j])
                totalpoilist_index.remove(j) 
        totalroutelist1.append(randomroute_name1)  
    return totalroutelist1

top=1
userdf=pd.DataFrame()
for l in range(1,101):
    print(str(l)+":开始")
    PrefixSpanresult=pd.read_excel("路线得分//"+str(l)+".xlsx")#0.1
    V_usepre=test.loc[l-1,'人均消费']
    routenum=int(float(test.loc[l-1,'天数'][:-1]))
    llist=[]
    llist1=ast.literal_eval(test.loc[l-1,'路线'])
    for l1 in llist1:
        llist.append(llist1[l1])
    userdf.loc[l-1,'实际路线']=str(llist)
    
    uniqueroute_finallist=[]
    i=0
    while i<top:
        if i<=len(PrefixSpanresult)-1:
            totalpoilist=X_POI(i)
            uniqueroute=LUWEN(routenum,totalpoilist)
            while len(uniqueroute)<routenum:
                uniqueroute.append([])
            uniqueroute_finallist.append(uniqueroute)
            i=i+1
        else:
            uniqueroute=[]
            for j in range(routenum):
                uniqueroute.append([])
            uniqueroute_finallist.append(uniqueroute)
            i=i+1 
    
    userdf.loc[l-1,'预测路线']=str(uniqueroute_finallist)
    print(str(l)+":预测路线")
    print(uniqueroute_finallist)
    
    print(str(l)+":计算命中率")
    ac=[oq for item in llist for oq in item]#真实路线
    a1=0
    for d in ac:
        for dd in range(0,routenum):#旅行天数
            for ddd in range(top):
                if (d in uniqueroute_finallist[ddd][dd]) and (d in llist[dd]):
                    a1=a1+1
                    break
    userdf.loc[l-1,'考虑天数的命中率']=a1/len(set(ac))
    
    print(str(l)+":结束")
    print(userdf)
    print(userdf.loc[:,"预测路线"].values)
userdf.to_csv("TOP1-Lenwen模型.csv")#0.1


top=4
userdf=pd.DataFrame()
for l in range(1,101):
    print(str(l)+":开始")
    PrefixSpanresult=pd.read_excel("路线得分//"+str(l)+".xlsx")#0.1
    V_usepre=test.loc[l-1,'人均消费']
    routenum=int(float(test.loc[l-1,'天数'][:-1]))
    llist=[]
    llist1=ast.literal_eval(test.loc[l-1,'路线'])
    for l1 in llist1:
        llist.append(llist1[l1])
    userdf.loc[l-1,'实际路线']=str(llist)
    
    uniqueroute_finallist=[]
    i=0
    while i<top:
        if i<=len(PrefixSpanresult)-1:
            totalpoilist=X_POI(i)
            uniqueroute=LUWEN(routenum,totalpoilist)
            while len(uniqueroute)<routenum:
                uniqueroute.append([])
            uniqueroute_finallist.append(uniqueroute)
            i=i+1
        else:
            uniqueroute=[]
            for j in range(routenum):
                uniqueroute.append([])
            uniqueroute_finallist.append(uniqueroute)
            i=i+1 
    
    userdf.loc[l-1,'预测路线']=str(uniqueroute_finallist)
    print(str(l)+":预测路线")
    print(uniqueroute_finallist)
    
    print(str(l)+":计算命中率")
    ac=[oq for item in llist for oq in item]
    a1=0
    for d in ac:
        for dd in range(0,routenum):
            for ddd in range(top):
                if (d in uniqueroute_finallist[ddd][dd]) and (d in llist[dd]):
                    a1=a1+1
                    break
    userdf.loc[l-1,'考虑天数的命中率']=a1/len(set(ac))
    
    print(str(l)+":结束")
    print(userdf)
    print(userdf.loc[:,"预测路线"].values)
userdf.to_csv("TOP4-Lenwen模型.csv")#0.1