__author__ = 'jruffio'

import numpy as np
import os
import sys
import glob
import time
import datetime

print("coucou")
#os.system("module load python/3.6.1")


OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
# foldername = "HR_8799_b"
N=0
for foldername in ["HR_8799_c","kap_And"]:#["HR_8799_b","HR_8799_c","HR_8799_d"]:#["kap_And","HR_8799_b","HR_8799_c","HR_8799_d"]:
    year = "2*"
    #year = "201*" 
    #year = "20171103"
    #year = "20100715"
    #year = "20110723"
    #year = "20101104"
    #year = "20150720"
    reductionname = "reduced_jb"
    #filenamefilter = "s*_a*001_tlc_Kbb_020.fits"
    # filenamefilter = "s*Kbb*.fits"
    for filenamefilter in ["s*Kbb*.fits","s*Hbb*.fits"]:
        #filenamefilter = "s101104_a03*001_Hbb_020.fits"
        planet_search = 1 # If True, pixel resolution entire FOV, otherwise centroid
        debug_paras = 1 # If True, fast reduction
        fakes = 0
        # planet_model_string = "'CO'"#"'model'"#"'CO'"#"'CO2'"#H2O#CH4#"'model'"
        #for planet_model_string in ["'joint'"]:#["'CO'","'CO2'","'H2O'","'CH4'"]:
        for planet_model_string in ["'model'"]:
            for resnumbasis in [0,1,10]:#np.arange(0,20):#[5,4,3,2,1]:
                filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
                filelist.sort()
                for fileid, filename in enumerate(filelist):
                    print(fileid,filename)
                    #continue

                    inputdir = os.path.dirname(filename)

                    #script = "~/OSIRIS/20180905_defcen_parallelized_osiris.py"
                    # script = "~/OSIRIS/20180909_2ndorderpoly_parallelized_osiris.py"
                    # script = "~/OSIRIS/osirisextract/parallelized_osiris.py"
                    script = "~/OSIRIS/osirisextract/reduce_HPFonly_diagcov_resmodel_v2.py"
                    #script =  "~/OSIRIS/osirisextract/reduce_HPFonly_diagcov_makemodel.py"
                    #script = "~/OSIRIS/osirisextract/classic_CCF.py"

                    logdir = os.path.join(inputdir,"sherlock","logs")
                    if not os.path.exists(logdir):
                        os.makedirs(logdir)
                    now = "{date:%Y%m%d_%H%M%S}_".format(date=datetime.datetime.now())
                    outfile = os.path.join(logdir,now+os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits","{0}.out".format(planet_search)))
                    errfile = os.path.join(logdir,now+os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits","{0}.err".format(planet_search)))
                    
                    outputdir = os.path.join(inputdir,"sherlock","20191202_newresmodel")#"20191120_newres_RV")#"20191120_newresmodel") # _RVsearch _single_slice
                    if 1 and not fakes and len(glob.glob(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}.fits".format(resnumbasis))))) >= 1:
                        print("skip"+filename)
                        continue
                    if 1 and fakes and len(glob.glob(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}_fakes.fits".format(resnumbasis))))) >= 1:
                        print("skip"+filename)
                        continue
                    #print(filename)
                    #continue
                    numthreads = 5
                    bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=2-00:00:00 --mem=20G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python3 ' + script
                    params = ' {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}"'.format(OSIRISDATA,inputdir,outputdir,filename,numthreads,planet_search,planet_model_string,debug_paras,resnumbasis,fakes)

                    N+=1
                    print(N,bsub_str+params)
                    #continue
                    #exit()
                    bsub_out = os.popen(bsub_str + params).read()
                    print(bsub_out)
                    #jobid_list.append(bsub_out.split(" ")[-1].strip('\n'))

                    #exit()
                    time.sleep(2)
                    #exit()
