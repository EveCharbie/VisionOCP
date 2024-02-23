#!/usr/bin/sudo bash


for (( i=0; i<100; i++ ))
do
	python 42_IOCP.py
	sudo pkill python
done



