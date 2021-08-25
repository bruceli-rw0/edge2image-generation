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
		--config config/pix2pixHD.yaml \
		--do_train \
		--do_eval \
		--num_train 3 \
		--num_eval 3 \

.PHONY: test-all
test-all:
	clear
	./env/bin/python3 -m generator \
		--config config/pix2pix.yaml \
		--do_train \
		--do_eval \
		--num_train 3 \
		--num_eval 3
	
	./env/bin/python3 -m generator \
		--config config/pix2pixHD.yaml \
		--do_train \
		--do_eval \
		--num_train 3 \
		--num_eval 3

.PHONY: download
download:
	if test -d datasets; \
	then echo "datasets already exist."; \
	else mkdir datasets; \
	fi

	./env/bin/gdown https://drive.google.com/uc?id=1kXmL9NQNLJcXJKe1xNrCCsUpYR1ZUFcz -O datasets/kaggle_landscape.zip

.PHONY: edges
edges:
	./env/bin/python3 hed/python/batch_hed.py \
		--caffe_root .. \
		--caffemodel hed/examples/hed_pretrained_bsds.caffemodel \
		--prototxt hed/examples/deploy.prototxt \
		--images_dir datasets/lhq_256/samples/images \
		--hed_mat_dir datasets/lhq_256/samples/edges

.PHONY: deploy
deploy:
	if test -d _checkpoints; \
	then echo "_checkpoints already exist."; \
	else mkdir _checkpoints; \
	fi

	if test -d _checkpoints/pix2pixHD-2021-08-13-04-02-59; \
	then echo "_checkpoints/pix2pixHD-2021-08-13-04-02-59 already exist."; \
	else mkdir _checkpoints/pix2pixHD-2021-08-13-04-02-59; \
	fi

	./env/bin/gdown https://drive.google.com/uc?id=1v2BXpN0F1I91-aN5wZzsKGvq1HNiO_Sn -O _checkpoints/pix2pixHD-2021-08-13-04-02-59/config.yaml
	./env/bin/gdown https://drive.google.com/uc?id=1hA5Fx5LdX6igeU0gN4UA_T0Agy29GE1K -O _checkpoints/pix2pixHD-2021-08-13-04-02-59/100_net_G.pth

.PHONY: clean
clean:
	rm -rf _checkpoints _results _loggings _metrics
