#!/bin/bash

cur=`ls -d /mnt/d/skycam/20* | tail -2 | head -1 | cut -d'/' -f5`

/home/skycam/skycam-utils/scripts/make_movies.sh $cur
