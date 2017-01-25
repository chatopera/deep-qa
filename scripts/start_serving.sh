#! /bin/bash 
###########################################
# Start the serving model
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/../deepqa2/serve
python manage.py makemigrations
python manage.py migrate
python manage.py runserver 0.0.0.0:8000