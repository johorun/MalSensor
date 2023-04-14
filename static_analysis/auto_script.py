import os 
import subprocess
import sys
from get_call_graph import *
import time


def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains malware.', required=True, type=str)
    parser.add_argument('-i', '--idapro', help='The path of IDA Pro', required=True, type=str)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    ida_path = args.idapro
    work_dir = os.path.abspath('.')
    #pefile_dir = os.path.join(work_dir, 'pefile')
    pefile_dir = args.dir
    script_path = os.path.join(work_dir, 'apicount.py')

    pefile_list = os.listdir(pefile_dir)
    pefile_num = len(pefile_list)
    print(pefile_num)
    for file in pefile_list:
        # sys.stdout.flush()
        # subprocess.call(cmd_cd, shell=True)
        #if file.endswith('dll') or file.endswith('exe'):
            # p = subprocess.Popen((cmd_str))
        cmd_ida = '{} -Lida.log -c -A -S{} {}'.format(ida_path,script_path, os.path.join(pefile_dir, file))
            #sys.stdout.flush()
    #   tic = time.time()
        subprocess.call(cmd_ida, shell=True)
    #    tic = time.time() - tic
    #   with open('D:\\IDAPro\\workspace\\e.txt', 'w+') as f:
    #      f.write(str(tic))
        #print(cmd_ida)
            #p.wait()
        #use_ida_to_get_call_graph('D:\IDAPro\workspace\output\\')

if __name__ == '__main__':
    main()
