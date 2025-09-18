
from io import StringIO
import re
import pandas as pd


# df = pd.read_csv('data/train.csv')
# ser = df.head(1).squeeze()
# raw = ser.to_csv(index=False, header=False)

raw = '''
0
工业/危化品类（现场）—2016版
（二）电气安全
6、移动用电产品、电动工具及照明
1、移动使用的用电产品和I类电动工具的绝缘线，必须采用三芯(单相)或四芯(三相)多股铜芯橡套软线。
"使用移动手动电动工具,外接线绝缘皮破损,应停止使用."
0
'''


ser: pd.Series = pd.read_csv(StringIO(raw), header=None).squeeze()


lv1, lv2, lv3, lv4 = ser[1:1+4]


lv1.split('（')[0],\
lv2.split('）')[-1], \
re.split(r'[0-9]、', lv3)[-1], \
re.split(r'[0-9]、', lv4)[-1]