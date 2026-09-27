import os,time,json,traceback
from data import ROOT,dump,prepare_dev,prepare_openood,prepare_new

def main():
    stages=[('dev',prepare_dev),('cifar',lambda:prepare_new('cifar')),('cub',lambda:prepare_new('cub')),('openood',prepare_openood)]
    for name,fn in stages:
        status=ROOT/'status'/f'prepare_{name}.json'
        dump(status,{'phase':'running','pid':os.getpid(),'stage':name})
        try:
            fn();dump(status,{'phase':'complete','stage':name})
        except Exception as e:
            dump(status,{'phase':'failed','stage':name,'error':repr(e),'traceback':traceback.format_exc()})
            print(traceback.format_exc(),flush=True)
            # Independent datasets are still prepared; the queue reports dependent jobs individually.
            if name=='openood':raise
    from plans import make_plans
    make_plans();dump(ROOT/'status/preparation.json',{'phase':'complete'})

if __name__=='__main__':main()
