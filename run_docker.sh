WORKSPACE=/media/data/d-rise

docker run -it \
	--gpus all \
	--net host \
    -w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	yhsmiley/drise:v1
