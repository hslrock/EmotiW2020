import re
import pandas as pd
import os

def trans_arff2csv(file_in,file_out):
       
    first=True
    file_list=os.listdir(file_in)
    file_list=sorted(file_list)
    for i in (file_list):
        if i.endswith(".arff"):
            file_name=os.path.join(file_in,i)
            
            with open(file_name, 'r') as f:
                data=[]
                columns=[]

                data_flag = 0 
                for line in f:
                    if line[:2] == '@a':
                        # find indices
                        indices = [i for i, x in enumerate(line) if x == ' ']
                        columns.append(re.sub(r'^[\'\"]|[\'\"]$|\\+', '', line[indices[0] + 1:indices[-1]]))
                    elif line[:2] == '@d':
                        data_flag = 1
                    elif data_flag == 1:
                        data.append(line)
            data=data[1].split(",")[2:-1]
            data.insert(0, os.path.splitext(os.path.splitext(i)[0])[0]+'.arff') 
            data=np.array([data])
            columns=columns[1:-1]

            columns.insert(0, "file_name") 
            
           # content = ','.join(columns) + '\n' + ''.join(data)
            if first:
                df1=pd.DataFrame(data=data,columns=columns)
                first=False
            else:
                df=pd.DataFrame(data=data,columns=columns)    
                df1=df1.append(df)
    df1.to_csv(file_out)
    return df1
