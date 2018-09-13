__author__ = 'jruffio'


import os
import sys
import glob
import time

print("coucou")

OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
if 1:
    foldername = "HR_8799_c"
    sep = 0.950
    telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
    template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
year = "*"
reductionname = "reduced_quinn"
filenamefilter = "s*_a*001_tlc_Kbb_020.fits"

filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
filelist.sort()
for filename in filelist:
    print(filename)
    #continue

    inputdir = os.path.dirname(filename)

    #script = "~/OSIRIS/20180905_defcen_parallelized_osiris.py"
    # script = "~/OSIRIS/20180909_2ndorderpoly_parallelized_osiris.py"
    script = "~/OSIRIS/osirisextract/parallelized_osiris.py"
    #script = "~/OSIRIS/osirisextract/classic_CCF.py"
    
    logdir = os.path.join(inputdir,"sherlock","logs")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    outfile = os.path.join(logdir,os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits",".out"))
    errfile = os.path.join(logdir,os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits",".err"))

    if 1:
        cenmode = "visu"
        outputdir = os.path.join(inputdir,"sherlock","polyfit_"+cenmode+"cen")
        numthreads = 20
        bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=0-12:00:00 --mem=120G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python ' + script
        params = ' {0} {1} {2} {3} {4} {5} {6} {7}"'.format(inputdir,outputdir,filename,telluric,template_spec,sep,numthreads,cenmode)
    else:
        outputdir = os.path.join(inputdir,"sherlock","medfilt_ccmap")
        numthreads = 1
        bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=0-01:00:00 --mem=10G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python ' + script
        params = ' {0} {1} {2} {3} {4} {5} {6}"'.format(inputdir,outputdir,filename,telluric,template_spec,sep,numthreads)

    print(bsub_str+params)
    bsub_out = os.popen(bsub_str + params).read()
    print(bsub_out)
    #jobid_list.append(bsub_out.split(" ")[-1].strip('\n'))
                    
    time.sleep(2)
    #exit()
