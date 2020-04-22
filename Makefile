.PHONY : clean run
WORKDIR = /home/ubuntu/work

#REPONAME="python"
#IMAGENAME="3.7.7-stretch"
#IMAGENAME=1.3-cuda10.1-cudnn7-devel
REPONAME = pytorch/pytorch
IMAGENAME = latest
#IMAGENAME = pytorch

download: 
	wget http://people.cs.nctu.edu.tw/~ysl/NCHC/download.sh
	chmod +x ./download.sh && ./download.sh
clean:
	rm -rf *.sh* ../work/en*tar* ../work/env_file ../work/*.sh*  
run: 
	
	# instantiate container
	docker pull ${REPONAME}:${IMAGENAME}
	docker run --rm --name test_container --runtime=nvidia -e CUDA_VISIBLE_DEVICES=1 --privileged -i -t --env-file ${WORKDIR}/env_file -v ${WORKDIR}:/mnt ${REPONAME}:${IMAGENAME}   bash -c "nvidia-smi; sh /mnt/container.sh"

