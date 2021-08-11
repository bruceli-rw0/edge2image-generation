.PHONY: setup
setup:
	if test -d env; \
	then echo "env already exist; update requirements only"; \
		./env/bin/python3 -m pip install -r requirements.txt; \
	else python3 -m venv env; \
		./env/bin/python3 -m pip install --upgrade pip; \
		./env/bin/python3 -m pip install -r requirements.txt; \
	fi	

.PHONY: test
test:
	clear
	./env/bin/python3 -m generator \
		--config config/pix2pix.yaml \
		--model pix2pix \
		--init_type kaiming \
		--use_dropout \
		--do_train \
		--do_eval \
		--n_epochs 2 \
		--num_train 3 \
		--num_eval 3 \

.PHONY: download
download:
	if test -d datasets; \
	then echo "datasets already exist."; \
	else mkdir datasets; \
	fi

	./env/bin/gdown https://drive.google.com/uc?id=1kXmL9NQNLJcXJKe1xNrCCsUpYR1ZUFcz -O datasets/kaggle_landscape.zip

# .PHONY: download-data
# download-data:
# 	if test -d datasets; \
# 	then echo "datasets already exist."; \
# 	else mkdir datasets; \
# 	fi

# 	wget -N http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz \
# 		 -O ./datasets/edges2shoes.tar.gz
# 	tar -zxvf ./datasets/edges2shoes.tar.gz -C ./datasets
# 	rm ./datasets/edges2shoes.tar.gz

# 	wget -N http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2handbags.tar.gz \
# 		 -O ./datasets/edges2handbags.tar.gz
# 	tar -zxvf ./datasets/edges2handbags.tar.gz -C ./datasets
# 	rm ./datasets/edges2handbags.tar.gz

# 	wget -N https://s323sas.storage.yandex.net/rdisk/058881b5eb9f251e06f9df247641e97f6bf9cd4928dc92cb51f5729c6d011bbb/610b2d22/k1ojKg_NbA2N_apAt0UuncqCnIEI0W3zqYS4XRlzsiSVQ_Ewz_CkudfOUIlWgya0YpndP534HCMIvORn2vZ-DA==?uid=0&filename=LHQ1024_jpg.zip&disposition=attachment&hash=DdPkbTGqyTcQBiBiPVxYxm1SIXH4YDqIL/OpdJUI4NX%2B9U62sNXSvk1tPb7Q987eq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=application%2Fzip&owner_uid=179435506&fsize=12862357468&hid=c18adf597e16bef0888fb0d652f4daad&media_type=compressed&tknv=v2&rtoken=vrdBHCE9apqw&force_default=no&ycrid=na-82bf526eafed593823112eabd56854bc-downloader16e&ts=5c8c4c96c0c80&s=a2dbe3384268f12af9b9d56a20d4122ecaa4ce975d4cfb20efd05e87d8da0722&pb=U2FsdGVkX1_-eVWy9WD2K6OnszACxhDFxferUp-gysiCkMEYGx3PPD9WO1-gwzK0CZsXdF_knTPso4ea_csv0ZuaShnEb4tQ0YxhHV8uoEM \
# 		 -O ./datasets/lhq1024.tar.gz
# 	tar -zxvf ./datasets/lhq1024.tar.gz -C ./datasets
# 	rm ./datasets/lhq1024.tar.gz

# .PHONY: download-data-sample
# download-data-sample:
# 	if test -d datasets; \
# 	then echo "datasets already exist."; \
# 	else mkdir datasets; \
# 	fi

# 	wget -N http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz \
# 		 -O ./datasets/edges2shoes.tar.gz
# 	tar -zxvf ./datasets/edges2shoes.tar.gz -C ./datasets
# 	rm ./datasets/edges2shoes.tar.gz

.PHONY: edges
edges:
	./env/bin/python3 hed/python/batch_hed.py \
		--caffe_root .. \
		--caffemodel hed/examples/hed_pretrained_bsds.caffemodel \
		--prototxt hed/examples/deploy.prototxt \
		--images_dir datasets/lhq_256/samples/images \
		--hed_mat_dir datasets/lhq_256/samples/edges

.PHONY: clean
clean:
	rm -rf _checkpoints _results _loggings _metrics
