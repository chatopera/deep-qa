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
    docker run --name deepqa2  \
        -e "CHATBOT_SECRET_KEY=myfoobar" \
        -v $rootDir/save:/deepqa2/save \
        -v $rootDir/data:/deepqa2/data \
        $* samurais/deepqa2:latest \
        
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

