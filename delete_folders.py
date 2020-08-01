__author__ = 'jruffio'

exit()
from glob import glob
import shutil

dirlist = glob("/data/osiris_data/*/*/reduced_jb/sherlock/20190510_spec_esti/")
dirlist.sort()
for mydir in dirlist: #/data/osiris_data/HR_8799_b/20090722/reduced_jb/sherlock/20190416_no_persis_corr/
    print(mydir)
    # shutil.rmtree(mydir)
    # exit()