import pandas as pd
import numpy as np
import os


df_all = pd.read_csv('../input/get-clusterings/df_all_with_clusters.csv')
tr = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv',usecols=['ID_code','target'])


df = df_all.merge(tr,on='ID_code')


df.groupby('cluster_id2')['target'].agg(['size','mean'])


df.groupby('cluster_id3')['target'].agg(['size','mean'])


df.groupby('cluster_id4')['target'].agg(['size','mean'])


df.groupby('cluster_id5')['target'].agg(['size','mean'])


df.groupby('cluster_id6')['target'].agg(['size','mean'])


df.groupby('cluster_id7')['target'].agg(['size','mean'])


df.groupby('cluster_id7')['target'].agg(['size','mean'])


df.groupby('cluster_id8')['target'].agg(['size','mean'])


df.groupby('cluster_id9')['target'].agg(['size','mean'])


df.groupby('cluster_id10')['target'].agg(['size','mean'])


df.groupby('cluster_id11')['target'].agg(['size','mean'])


df.groupby('cluster_id12')['target'].agg(['size','mean'])


df.groupby('cluster_id13')['target'].agg(['size','mean'])


df.groupby('cluster_id14')['target'].agg(['size','mean'])


df.groupby('cluster_id15')['target'].agg(['size','mean'])


df.groupby('cluster_id16')['target'].agg(['size','mean'])


df.groupby('cluster_id17')['target'].agg(['size','mean'])


df.groupby('cluster_id18')['target'].agg(['size','mean'])


df.groupby('cluster_id19')['target'].agg(['size','mean'])


df.groupby('cluster_id20')['target'].agg(['size','mean'])


df.groupby('cluster_id21')['target'].agg(['size','mean'])


df.groupby('cluster_id22')['target'].agg(['size','mean'])


df.groupby('cluster_id23')['target'].agg(['size','mean'])


df.groupby('cluster_id24')['target'].agg(['size','mean'])


df.groupby('cluster_id25')['target'].agg(['size','mean'])


df.groupby('cluster_id26')['target'].agg(['size','mean'])


df.groupby('cluster_id27')['target'].agg(['size','mean'])


df.groupby('cluster_id28')['target'].agg(['size','mean'])


df.groupby('cluster_id29')['target'].agg(['size','mean'])


df.groupby('cluster_id30')['target'].agg(['size','mean'])



