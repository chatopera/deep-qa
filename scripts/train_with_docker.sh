#! /bin/bash 
###########################################
# Start train with docker
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
rootDir=$baseDir/..

# functions
function printUsage(){
    echo "Usage:"
    echo "$0 -d # This would run the docker in detached mode."
    echo "$0 -t # This would run the docker in not-attached mode."
}

function main() {
    cd $rootDir
    nvidia-docker run --name deepqa2  \
        -v $rootDir/save:/deepqa2/save \
        -v $rootDir/logs:/deepqa2/logs \
        -v $rootDir/data:/deepqa2/data \
        -v $rootDir/config.ini:/deepqa2/config.ini \
        -v $rootDir/start_training_docker.sh:/deepqa2/start_training_docker.sh \
        $* samurais/deepqa2:latest \
        "./start_training_docker.sh"
}

# main
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
if [ "$#" -eq  "0" ]
then
    printUsage
elif [ "$*" = "-t" ]
then
    main -t -i --rm
elif [ "$*" = "-d" ]
then
    main -d
else
    printUsage
fi
