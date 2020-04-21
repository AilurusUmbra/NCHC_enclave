.PHONY : clean run
WORKDIR = /home/ubuntu/work

#IMAGENAME="ubuntu"
#IMAGENAME="pytorch:1.3-cuda10.1-cudnn7-devel"
REPONAME = pytorch
IMAGENAME = pytorch

download: 
			wget http://people.cs.nctu.edu.tw/~ysl/NCHC/download.sh
			chmod +x ./download.sh && ./download.sh
clean:
			rm -rf *.sh* ../work/en*tar ../work/env_file ../work/*.sh 
run: 
			
			# instantiate container
			docker pull ${REPONAME}/${IMAGENAME}
			docker run --rm --name test_container --runtime=nvidia --privileged -i -t --env-file ${WORKDIR}/env_file -v ${WORKDIR}:/mnt ${REPONAME}/${IMAGENAME}   bash -c "sh /mnt/container.sh"
