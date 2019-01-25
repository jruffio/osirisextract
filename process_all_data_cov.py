__author__ = 'jruffio'

import numpy as np
import os
import sys
import glob
import time

#print("coucou")
#os.system("module load python/3.6.1")


OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
if 1:
    foldername = "HR_8799_c"
    sep = 0.950
    #telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
    #template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
year = "*"
# year = "20100715"
# year = "20110723"
reductionname = "reduced_jb"
#filenamefilter = "s*_a*001_tlc_Kbb_020.fits"
#filenamefilter = "s*_020.fits"
# filenamefilter = "s110723_a026001_Kbb_020.fits"
filenamefilter = "s100715_a010001_Kbb_020.fits"
planet_search = 1 # pixel resolution entire FOV
# planet_search = 0 # centroid only

filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
filelist.sort()
# filename = filelist[0]
# for lcorr in np.arange(0.1,2.0,0.1):
for filename in filelist:
    lcorr = 0
    print(filename)
    #continue

    inputdir = os.path.dirname(filename)

    #script = "~/OSIRIS/20180905_defcen_parallelized_osiris.py"
    # script = "~/OSIRIS/20180909_2ndorderpoly_parallelized_osiris.py"
    # script = "~/OSIRIS/osirisextract/parallelized_osiris.py"
    script = "~/OSIRIS/osirisextract/reduce_HPFonly_cov.py"
    #script = "~/OSIRIS/osirisextract/classic_CCF.py"
    
    logdir = os.path.join(inputdir,"sherlock","logs")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    outfile = os.path.join(logdir,os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits","_cov.out"))
    errfile = os.path.join(logdir,os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits","_cov.err"))

    outputdir = os.path.join(inputdir,"sherlock","20190117_HPFonly_cov")
    numthreads = 16
    bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=2-0:00:00 --mem=60G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python3 ' + script
    params = ' {0} {1} {2} {3} {4} {5}"'.format(inputdir,outputdir,filename,numthreads,planet_search,lcorr)
    # if 1:
    #     cenmode = "visu"
    #     outputdir = os.path.join(inputdir,"sherlock","polyfit_"+cenmode+"cen")
    #     numthreads = 20
    #     bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=0-12:00:00 --mem=120G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python ' + script
    #     params = ' {0} {1} {2} {3} {4} {5} {6} {7}"'.format(inputdir,outputdir,filename,telluric,template_spec,sep,numthreads,cenmode)
    # else:
    #     outputdir = os.path.join(inputdir,"sherlock","medfilt_ccmap")
    #     numthreads = 1
    #     bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=0-01:00:00 --mem=10G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python ' + script
    #     params = ' {0} {1} {2} {3} {4} {5} {6}"'.format(inputdir,outputdir,filename,telluric,template_spec,sep,numthreads)

    print(bsub_str+params)
    bsub_out = os.popen(bsub_str + params).read()
    print(bsub_out)
    #jobid_list.append(bsub_out.split(" ")[-1].strip('\n'))

    exit()
    time.sleep(2)
