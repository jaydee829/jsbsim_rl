#!/bin/bash
echo "Starting VNC (josh: $AUX_CMD)"
/dockerstartup/vnc_startup.sh  $AUX_CMD

cd /home/jsbsim
#echo "Starting jupyter"
#jupyter notebook --ip 0.0.0.0 --no-browser
