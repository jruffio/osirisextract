__author__ = 'jruffio'

import numpy as np
import os
import sys
import glob
import time
import datetime

print("coucou")
#os.system("module load python/3.6.1")

OSIRISDATA = "/data/osiris_data/"
# OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
N=0
# /home/anaconda3/bin/python3
# /home/sda/jruffio/pyOSIRIS/osirisextract/mcmc_atm_charac.py
for foldername in ["HR_8799_b","HR_8799_c","HR_8799_d"]:#["kap_And","HR_8799_b","HR_8799_c","HR_8799_d"]:
    year = "*"
    reductionname = "reduced_jb"
    numthreads = 4
    small = 1
    modelfolder = "20200309_model"
    outfolder = os.path.join("sherlock","20200312_travisgridpost")
    for filenamefilter in ["s*Kbb*.fits"]:#,"s*Hbb*.fits"]:
        gridmodel = "sonora"
        for planet_model_string in ["'model'"]:
            for resnumbasis in [10]:
                filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
                filelist.sort()
                for fileid, filename in enumerate(filelist):
                    print(filename)
                    #continue
                    tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_corrwvs.fits"))
                    if len(glob.glob(tmpfilename))!=1:
                        print("No data on "+filename)
                        continue

                    inputdir = os.path.dirname(filename)

                    script = "~/OSIRIS/osirisextract/mcmc_atm_charac.py"

                    logdir = os.path.join(inputdir,"sherlock","logs")
                    if not os.path.exists(logdir):
                        os.makedirs(logdir)
                    now = "{date:%Y%m%d_%H%M%S}_".format(date=datetime.datetime.now())
                    outfile = os.path.join(logdir,now+os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits",".out"))
                    errfile = os.path.join(logdir,now+os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits",".err"))

                    # if 0 and len(glob.glob(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}.fits".format(resnumbasis))))) >= 1:
                    #     print("skip"+filename)
                    #     continue
                    bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=2-00:00:00 --mem=10G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python3 ' + script
                    params = ' {0} {1} {2} {3} {4} {5} {6} {7}"'.format(OSIRISDATA,modelfolder,outfolder,filename,numthreads,gridmodel,resnumbasis,small)

                    # "/data/osiris_data/"
                    # "20200309_model"
                    # "sherlock/20200312_modeltest"
                    # "/data/osiris_data/HR_8799_c/20100715/reduced_jb/s100715_a010001_Kbb_020.fits"
                    # 5
                    # "/data/osiris_data/hr8799b_modelgrid"
                    # 10
                    # 1
                    N+=1
                    print(N,bsub_str+params)
                    #continue
                    #exit()
                    # bsub_out = os.popen(bsub_str + params).read()
                    # print(bsub_out)
                    #jobid_list.append(bsub_out.split(" ")[-1].strip('\n'))

                    #exit()
                    # time.sleep(2)
                    #exit()
