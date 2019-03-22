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
# IFSfilter = "Jbb"
IFSfilter = "Hbb"
#IFSfilter = "Kbb"
# refstar_name = "HD_210501"
refstar_name = "HR_8799"
cutoff = 5

IFSfilter_list = ["Hbb"]#["Jbb","Hbb","Kbb"]
#refstar_name_list = ["HD_210501"]
refstar_name_list = ["HD_210501","HIP_1123","HR_8799","BD+14_4774","HD_7215","HIP_18717"]

for IFSfilter in IFSfilter_list:
    for refstar_name in refstar_name_list:

        filename_filter = "s*"+IFSfilter+"*020_psfs_repaired_spec_v2.fits"
        filelist = glob.glob(os.path.join(OSIRISDATA,"HR_8799_*","*","reduced_telluric_jb",refstar_name,filename_filter))
        filename_filter = "ao_off_s*"+IFSfilter+"*020_spec_v2.fits"
        filelist.extend(glob.glob(os.path.join(OSIRISDATA,"HR_8799_*","*","reduced_telluric_jb",refstar_name,filename_filter)))
        filelist.sort()

        for filename in filelist:
            print(filename)
            #continue

            inputdir = os.path.dirname(filename)
            script = "~/OSIRIS/osirisextract/charac_reference_stars.py"

            logdir = os.path.join(inputdir,"sherlock_logs")
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            now = "{date:%Y%m%d_%H%M%S}_".format(date=datetime.datetime.now())
            outfile = os.path.join(logdir,now+os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits",".out"))
            errfile = os.path.join(logdir,now+os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits",".err"))

            if 1 and len(glob.glob(os.path.join(filename.replace(".fits","_cutoff{0}_transmission.fits".format(cutoff))))) >= 1:
                print("skip"+filename)
                continue
            #continue
            numthreads = 16
            bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=2-0:00:00 --mem=60G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python3 ' + script
            params = ' {0} {1} {2} {3} {4} {5}"'.format(OSIRISDATA,filename,refstar_name,IFSfilter,numthreads,cutoff)
            # OSIRISDATA = sys.argv[1]
            # spec_filename = sys.argv[2]
            # refstar_name = sys.argv[3]
            # IFSfilter = sys.argv[4]
            # numthreads = sys.argv[5]
            # cutoff = sys.argv[6]

            print(bsub_str+params)
            #exit()
            bsub_out = os.popen(bsub_str + params).read()
            print(bsub_out)
            #jobid_list.append(bsub_out.split(" ")[-1].strip('\n'))

            #exit()
            time.sleep(2)
