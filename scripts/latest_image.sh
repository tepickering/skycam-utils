#!/bin/bash

scp /var/www/html/skycam/ImageLastFTP_AllSKY.jpg tim@ops.mmto.arizona.edu:/var/www/html/new_skycam/latest_image.jpg

sleep 30

scp /var/www/html/skycam/ImageLastFTP_AllSKY.jpg tim@ops.mmto.arizona.edu:/var/www/html/new_skycam/latest_image.jpg
