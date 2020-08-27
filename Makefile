PYTHON=python
#PYTHON=path to your python installation
path=examples/
recursive=false

make:
	@echo Installing pyross geo...
	${PYTHON} setup.py install
	@echo adding githook...
	cp .githooks/pre-push .git/hooks/
	chmod +x .git/hooks/pre-push

clean-local:
	@echo removing local compiled files
	rm pyrossgeo/*.c
	rm pyrossgeo/*.html
	rm pyrossgeo/*.so
	rm pyrossgeo/*.cpp
	 
clean:
	@echo removing all compiled files
	${PYTHON} setup.py clean
	rm pyrossgeo/*.c
	rm pyrossgeo/*.html
	rm pyrossgeo/*.so
	rm pyrossgeo/*.cpp
	rm -rf pyrossgeo/__pychache__/
	rm -rf build
	
env:
	@echo creating conda environment...
	conda env create --file environment.yml
	@echo use make to install pyross geo

test:
	@echo not implemented yet!	
	#@echo testing pyross...
	#cd tests && python quick_test.py

nbtest:
	@echo testing example notebooks...
	@echo test $(path)
	cd tests && python notebook_test.py --path $(path) --recursive $(recursive)

pypitest:
	@echo testing pystokes...
	python setup.py sdist bdist_wheel
	python -m twine upload --repository testpypi dist/*

pypi:
	@echo testing pystokes...
	python setup.py sdist bdist_wheel	
	python -m twine upload dist/*
