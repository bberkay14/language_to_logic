from path import Path
import zipfile
import urllib.request

Path('/Users/bilgeberkay/Desktop/lang2logic').mkdir_p()

for model_name in ('seq2seq','seq2tree'):
    for data_name in ('jobqueries','geoqueries','atis'):
        fn = '%s_%s.zip' % (model_name, data_name)
        link = 'http://dong.li/lang2logic/' + fn
        with open('/Users/bilgeberkay/Desktop/lang2logic/' + fn, 'wb') as f_out:
            f_out.write(urllib.request.urlopen(link).read())
        with zipfile.ZipFile('/Users/bilgeberkay/Desktop/lang2logic/' + fn) as zf:
            zf.extractall('./%s/%s/data/' % (model_name, data_name))
            
Path('/Users/bilgeberkay/Desktop/lang2logic').rmtree()