__author__ = 'jruffio'


import os
import sys
import glob
import time
import datetime

print("coucou")
#os.system("module load python/3.6.1")


OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
uservs = False
for planet in ["b","c","d","e"]:
    astrometry_DATADIR = os.path.join(OSIRISDATA,"astrometry")
    if uservs and (planet == "b" or planet =="c"):
        filename = "{0}/HR8799{1}_rvs.csv".format(astrometry_DATADIR,planet)
    else:
        filename = "{0}/HR8799{1}.csv".format(astrometry_DATADIR,planet)

    script = "~/OSIRIS/osirisextract/orbit_fit.py"

    logdir = os.path.join(astrometry_DATADIR,"logs")
    now = "{date:%Y%m%d_%H%M%S}_".format(date=datetime.datetime.now())
    outfile = os.path.join(logdir,now+os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits",".out"))
    errfile = os.path.join(logdir,now+os.path.basename(script).replace(".py","")+"_"+os.path.basename(filename).replace(".fits",".err"))

    num_temps = 20
    num_walkers = 100
    total_orbits = 20000 # number of steps x number of walkers (at lowest temperature)
    burn_steps = 100 # steps to burn in per walker
    thin = 2 # only save every 2nd step
    numthreads = 10
    suffix = "sherlock"

    # osiris_data_dir = sys.argv[1]
    # filename = sys.argv[2]
    # planet = sys.argv[3]
    # num_temps = int(sys.argv[4])
    # num_walkers = int(sys.argv[5])
    # total_orbits = int(sys.argv[6]) # number of steps x number of walkers (at lowest temperature)
    # burn_steps = int(sys.argv[7]) # steps to burn in per walker
    # thin = int(sys.argv[8]) # only save every 2nd step
    # num_threads = int(sys.argv[9]) # or a different number if you prefer
    # suffix = sys.argv[10]
    bsub_str= 'sbatch --partition=hns,owners,iric --qos=normal --time=2-00:00:00 --mem=20G --output='+outfile+' --error='+errfile+' --nodes=1 --ntasks-per-node='+str(numthreads)+' --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="python3 ' + script
    params = ' {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}"'.format(OSIRISDATA,filename,planet,num_temps,num_walkers,total_orbits,burn_steps,thin,numthreads,suffix)

    print(bsub_str+params)
    exit()
    bsub_out = os.popen(bsub_str + params).read()
    print(bsub_out)
    #jobid_list.append(bsub_out.split(" ")[-1].strip('\n'))

    #exit()
    time.sleep(2)