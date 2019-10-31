__author__ = 'jruffio'


import os
import sys
import glob
import time
import datetime

print("coucou")
#os.system("module load python/3.6.1")


#OSIRISDATA = "/data/osiris_data/"
OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
for k in range(10):
    for l in range(26):
        logdir = os.path.join(OSIRISDATA,"stellar_fits","sherlock_logs")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        now = "{date:%Y%m%d_%H%M%S}_".format(date=datetime.datetime.now())
        outfile = os.path.join(logdir,now+"_log_{0}_{1}.out".format(k,l))
        errfile = os.path.join(logdir,now+"_log_{0}_{1}.err".format(k,l))
        #continue
        numthreads = 16
        bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=2-0:00:00 --mem=60G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python3 ' + script
        params = ' {0} {1} "'.format(k,l)

        print(bsub_str+params)
        #exit()
        bsub_out = os.popen(bsub_str + params).read()
        print(bsub_out)
        #jobid_list.append(bsub_out.split(" ")[-1].strip('\n'))

        #exit()
        time.sleep(2)
