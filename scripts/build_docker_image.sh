#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
rootDir=$baseDir/..
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $rootDir 
docker build -f Dockerfile.gpu -t samurais/deepqa2:latest .
# --force-rm=true 
