#!/bin/bash

###############################################################################################################

python="python3"
libs="torchvision torch numpy scipy jupyter pandas matplotlib tqdm sklearnseaborn"
environments_dir=$HOME/environments
ip=95.213.170.234
port=25400
install_venv=false

###############################################################################################################

usage="Usage: $PROG [options]\n\n
Options:\n
  --help\t\t\tPrint this message and exit\n
  --python\t\tPython version (python2, python3, ect) (default=$python).\n
  --libs\t\t\tPython libs to install separated by space (default=$libs).\n
  --environments-dir\tEnvironements directory (default=$environments_dir).\n
  --host\t\t\tHost.\n
  --port\t\t\tPort.\n
  --install-venv\t\tInstall virtualenv (default=$install_venv).\n
";

###############################################################################################################

while [ $# -gt 0  ]; do
    case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
        --help) echo -e $usage; exit 0 ;;
        --python)
            shift; python="${1}"; shift ;;
        --libs)
            shift; libs="$1"; shift ;;
        --environments-dir)
            shift; environments_dir="$1"; shift ;;
	--ip)
	    shift; ip="$1"; shift ;;
        --port)
            shift; port="$1"; shift ;;
        --install-venv)
            install_venv=true; shift ;;
        -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
        *)   break ;;   # end of options: interpreted as num-leaves
    esac
done

#################################################################################################################

if [ ! -d "$environments_dir"  ]; then
    mkdir $environments_dir
fi

if [ -d "$environments_dir/$python" ]; then
    rm -r  "$environments_dir/$python"
fi

mkdir "$environments_dir/$python"

###############################################################################################################

if [[ $install_venv == true ]]; then
    sudo $python -m pip install --upgrade pip --user
    sudo $python -m pip install venv --user
fi

$python -m venv $environments_dir/$python
source $environments_dir/$python/bin/activate
pip3 install $libs
jupyter notebook --port=$port --ip $ip

##############################################################################################################t
